"""Copy-paste inference examples for the edge targets, generated from a run's own metadata so the
channels, classes, window and level scaling in the code are the ones the model was trained with.

Every example normalizes the window first, with the rule of
windowing.normalize_window_like_training: the models are meaningless on raw signals. The texts in
`VERIFICATION` say how far each example was actually tested; keep them true when changing one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MICRO_RESOLVER = {
    "ADD": "AddAdd", "BATCH_TO_SPACE_ND": "AddBatchToSpaceNd", "CONV_2D": "AddConv2D", "DEPTHWISE_CONV_2D": "AddDepthwiseConv2D", "DEQUANTIZE": "AddDequantize",
    "EXPAND_DIMS": "AddExpandDims", "FULLY_CONNECTED": "AddFullyConnected", "LOGISTIC": "AddLogistic", "MAX_POOL_2D": "AddMaxPool2D", "MEAN": "AddMean", "MUL": "AddMul",
    "PACK": "AddPack", "PAD": "AddPad", "QUANTIZE": "AddQuantize", "RELU": "AddRelu", "RESHAPE": "AddReshape", "SHAPE": "AddShape", "SOFTMAX": "AddSoftmax",
    "SPACE_TO_BATCH_ND": "AddSpaceToBatchNd", "SQUEEZE": "AddSqueeze", "STRIDED_SLICE": "AddStridedSlice",
}

VERIFICATION = {
    "esp32": "Not compiled: ESP-IDF is not part of this repository. Only the normalization block was tested, on the PC with gcc, against the Python rule.",
    "rpi5_python": "Tested on the PC with TensorFlow's interpreter, against the Keras model. Not run on a Raspberry Pi.",
    "rpi5_cpp": "Not built against TensorFlow Lite. Only the normalization block was tested, on the PC with gcc, against the Python rule.",
    "c2000": "Compiled and linked with cl2000 against the exported kit. read_sample() is a placeholder, so this example was not run on the board; the kit's own benchmark was.",
}


@dataclass
class ModelFacts:
    channels: list
    window: int
    classes: list = field(default_factory=list)
    horizon: int | None = None
    level_scaling: dict = field(default_factory=dict)
    calibration: dict | None = None
    tflite_ops: list = field(default_factory=list)

    @property
    def is_classifier(self) -> bool:
        return bool(self.classes)

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    @property
    def temperature(self) -> float:
        return float(self.calibration["temperature"]) if self.calibration else 1.0

    @property
    def abstain_below(self) -> float:
        return float(self.calibration["abstain_below"]) if self.calibration else 0.0


def facts_from_run(metrics: dict, config: dict, tflite_ops: list | None = None) -> ModelFacts:
    return ModelFacts(channels=list(metrics["channels"]), window=int(config["input_window"]), classes=list(metrics.get("classes") or []), horizon=config.get("horizon"),
                      level_scaling=dict(metrics.get("level_scaling") or {}), calibration=metrics.get("calibration"), tflite_ops=list(tflite_ops or []))


def _floats(values) -> str:
    return ", ".join(f"{float(v):.6g}f" if "." in f"{float(v):.6g}" or "e" in f"{float(v):.6g}" else f"{float(v):.1f}f" for v in values)


def _channel_comment(facts: ModelFacts) -> str:
    return ", ".join(facts.channels)


def normalization_block(facts: ModelFacts) -> str:
    """C / C++: the training pipeline's rule, for the ESP32 and Raspberry Pi C++ examples."""
    centers = [(facts.level_scaling.get(c) or (0.0, 0.0))[0] for c in facts.channels]
    scales = [(facts.level_scaling.get(c) or (0.0, 0.0))[1] for c in facts.channels]
    return f"""#define WINDOW_SAMPLES {facts.window}
#define NUM_CHANNELS {facts.n_channels}  // order: {_channel_comment(facts)}

// scale > 0 keeps the channel's absolute level; scale == 0 z-scores it with the window's statistics.
static const float LEVEL_CENTER[NUM_CHANNELS] = {{{_floats(centers)}}};
static const float LEVEL_SCALE[NUM_CHANNELS] = {{{_floats(scales)}}};

// Same rule as training. raw and out are (WINDOW_SAMPLES, NUM_CHANNELS), row by row.
static void normalize_window(const float *raw, float *out, float *mu, float *sigma) {{
  for (int c = 0; c < NUM_CHANNELS; c++) {{
    if (LEVEL_SCALE[c] > 0.0f) {{
      mu[c] = LEVEL_CENTER[c];
      sigma[c] = LEVEL_SCALE[c];
      for (int t = 0; t < WINDOW_SAMPLES; t++) out[t * NUM_CHANNELS + c] = (raw[t * NUM_CHANNELS + c] - mu[c]) / sigma[c];
      continue;
    }}
    double sum = 0.0, var = 0.0;
    for (int t = 0; t < WINDOW_SAMPLES; t++) sum += raw[t * NUM_CHANNELS + c];
    const double mean = sum / WINDOW_SAMPLES;
    for (int t = 0; t < WINDOW_SAMPLES; t++) {{
      const double d = raw[t * NUM_CHANNELS + c] - mean;
      var += d * d;
    }}
    const float m = (float)mean, s = (float)sqrt(var / WINDOW_SAMPLES);
    mu[c] = m;
    const int live = s > 1e-3f * (fabsf(m) + 1e-6f);  // a dead channel becomes zeros
    sigma[c] = live ? s : 1.0f;
    for (int t = 0; t < WINDOW_SAMPLES; t++) out[t * NUM_CHANNELS + c] = live ? (raw[t * NUM_CHANNELS + c] - m) / s : 0.0f;
  }}
}}"""


def _class_array(facts: ModelFacts) -> str:
    names = ", ".join(f'"{c}"' for c in facts.classes)
    return f"static const char *const CLASS_NAMES[{len(facts.classes)}] = {{{names}}};"


def esp32_example(facts: ModelFacts) -> str:
    ops = [o for o in facts.tflite_ops if not o.startswith("Flex")]
    known = [f"  resolver.{MICRO_RESOLVER[o]}();" for o in ops if o in MICRO_RESOLVER]
    unknown = [f"  // TODO: register {o}, it has no entry in this generator's op table." for o in ops if o not in MICRO_RESOLVER]
    resolver = "\n".join(known + unknown) if ops else "  // TODO: register the ops of your .tflite (no model.tflite was read when this was generated)."
    if facts.is_classifier:
        report = f"""    // Quantized output -> probabilities, calibrated with this run's temperature.
    float probs[{len(facts.classes)}], total = 0.0f;
    int best = 0;
    for (int k = 0; k < {len(facts.classes)}; k++) {{
      const float p = (output->data.int8[k] - output->params.zero_point) * output->params.scale;
      probs[k] = powf(p > 0.0f ? p : 0.0f, 1.0f / {facts.temperature:.6f}f);
      total += probs[k];
    }}
    for (int k = 0; k < {len(facts.classes)}; k++) {{
      probs[k] /= total;
      if (probs[k] > probs[best]) best = k;
    }}
    if (probs[best] < {facts.abstain_below:.6f}f) {{
      printf("uncertain (top %.2f)\\n", probs[best]);
    }} else {{
      printf("%s (%.2f)\\n", CLASS_NAMES[best], probs[best]);
    }}"""
        extra = _class_array(facts)
    else:
        report = f"""    // Quantized output -> normalized forecast -> physical units, channel by channel.
    for (int c = 0; c < NUM_CHANNELS; c++) {{
      const int last = ({facts.horizon} - 1) * NUM_CHANNELS + c;
      const float y = (output->data.int8[last] - output->params.zero_point) * output->params.scale;
      printf("channel %d, forecast at the horizon: %f\\n", c, y * sigma[c] + mu[c]);
    }}"""
        extra = ""
    return f"""// main/main.cc -- ESP-IDF + esp-tflite-micro. TFLite Micro has a C++ API only, so this one file
// is C++; the logic is written in plain C style.
#include <math.h>
#include <stdio.h>

#include "esp_rom_sys.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

extern "C" const unsigned char model_tflite[];  // main/model_data.c, generated with xxd

{normalization_block(facts)}

{extra}
// Tune it: start high, then use the value printed at startup.
static constexpr int kArenaSize = 96 * 1024;
alignas(16) static uint8_t tensor_arena[kArenaSize];
static float raw[WINDOW_SAMPLES * NUM_CHANNELS], normalized[WINDOW_SAMPLES * NUM_CHANNELS];

// TODO: replace with your ADC / sensor driver. Physical units, same units as in training.
static void read_sample(float *sample) {{
  for (int c = 0; c < NUM_CHANNELS; c++) sample[c] = 0.0f;
}}

extern "C" void app_main(void) {{
  const tflite::Model *model = tflite::GetModel(model_tflite);
  static tflite::MicroMutableOpResolver<{max(len(ops), 1)}> resolver;
{resolver}
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {{
    printf("AllocateTensors failed: increase kArenaSize\\n");
    return;
  }}
  printf("arena used: %u bytes\\n", (unsigned)interpreter.arena_used_bytes());
  TfLiteTensor *input = interpreter.input(0);
  TfLiteTensor *output = interpreter.output(0);

  for (;;) {{
    for (int t = 0; t < WINDOW_SAMPLES; t++) {{
      read_sample(&raw[t * NUM_CHANNELS]);
      esp_rom_delay_us(200);  // 5 kHz, the rate of the training data. Use a timer for real sampling.
    }}
    float mu[NUM_CHANNELS], sigma[NUM_CHANNELS];
    normalize_window(raw, normalized, mu, sigma);
    for (int i = 0; i < WINDOW_SAMPLES * NUM_CHANNELS; i++) {{
      const float q = normalized[i] / input->params.scale + input->params.zero_point;
      input->data.int8[i] = (int8_t)fmaxf(-128.0f, fminf(127.0f, roundf(q)));
    }}
    if (interpreter.Invoke() != kTfLiteOk) {{
      printf("Invoke failed\\n");
      return;
    }}
{report}
    vTaskDelay(1);  // lets the idle task feed the watchdog
  }}
}}
"""


def rpi_python_example(facts: ModelFacts) -> str:
    if facts.is_classifier:
        constants = f"CLASSES = {facts.classes!r}\nTEMPERATURE = {facts.temperature:.6f}\nABSTAIN_BELOW = {facts.abstain_below:.6f}\n"
        result = """    probs = np.clip(output[0], 1e-12, None) ** (1.0 / TEMPERATURE)
    probs /= probs.sum()
    best = int(np.argmax(probs))
    return ("uncertain" if probs[best] < ABSTAIN_BELOW else CLASSES[best]), probs"""
        doc = "(label, calibrated probabilities)"
        demo = '    label, probs = infer(raw)\n    print(label, probs)'
    else:
        constants = ""
        result = "    return output[0] * sigma + mu"
        doc = f"forecast of shape ({facts.horizon}, {facts.n_channels}), in physical units"
        demo = "    print(infer(raw)[-1])"
    return f"""# spade_infer.py -- pip install ai-edge-litert numpy
import numpy as np
from ai_edge_litert.interpreter import Interpreter

WINDOW_SAMPLES = {facts.window}
CHANNELS = {facts.channels!r}
LEVEL_SCALING = {facts.level_scaling!r}
{constants}
interpreter = Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_detail = interpreter.get_input_details()[0]
output_detail = interpreter.get_output_details()[0]


def normalize(raw):
    \"\"\"The rule used in training. raw: (WINDOW_SAMPLES, len(CHANNELS)), in physical units.\"\"\"
    out = np.zeros_like(raw)
    mu, sigma = raw.mean(axis=0), np.ones(raw.shape[1], dtype=np.float32)
    for c, name in enumerate(CHANNELS):
        w = raw[:, c]
        if name in LEVEL_SCALING:
            mu[c], sigma[c] = LEVEL_SCALING[name]
            out[:, c] = (w - mu[c]) / sigma[c]
        elif w.std() > 1e-3 * (abs(w.mean()) + 1e-6):  # a dead channel stays at zero
            sigma[c] = w.std()
            out[:, c] = (w - mu[c]) / sigma[c]
    return out, mu, sigma


def infer(raw):
    \"\"\"Returns {doc}.\"\"\"
    window, mu, sigma = normalize(np.asarray(raw, dtype=np.float32))
    interpreter.set_tensor(input_detail["index"], window[np.newaxis])
    interpreter.invoke()
    output = interpreter.get_tensor(output_detail["index"])
{result}


if __name__ == "__main__":
    # TODO: replace with a real window from your sensors.
    raw = np.random.default_rng(0).normal(size=(WINDOW_SAMPLES, len(CHANNELS)))
{demo}
"""


def rpi_cpp_example(facts: ModelFacts) -> str:
    if facts.is_classifier:
        n = len(facts.classes)
        extra = _class_array(facts)
        report = f"""  float probs[{n}], total = 0.0f;
  int best = 0;
  for (int k = 0; k < {n}; k++) {{
    probs[k] = powf(output[k] > 0.0f ? output[k] : 0.0f, 1.0f / {facts.temperature:.6f}f);
    total += probs[k];
  }}
  for (int k = 0; k < {n}; k++) {{
    probs[k] /= total;
    if (probs[k] > probs[best]) best = k;
  }}
  if (probs[best] < {facts.abstain_below:.6f}f) {{
    std::printf("uncertain (top %.2f)\\n", probs[best]);
  }} else {{
    std::printf("%s (%.2f)\\n", CLASS_NAMES[best], probs[best]);
  }}"""
    else:
        extra = ""
        report = f"""  for (int c = 0; c < NUM_CHANNELS; c++) {{
    const float y = output[({facts.horizon} - 1) * NUM_CHANNELS + c];
    std::printf("channel %d, forecast at the horizon: %f\\n", c, y * sigma[c] + mu[c]);
  }}"""
    return f"""// main.cpp -- TensorFlow Lite C++ API, on a Raspberry Pi running Linux.
#include <math.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"

{normalization_block(facts)}

{extra}
// TODO: replace with a real window from your sensors. Physical units, same units as in training.
static void read_window(std::vector<float> &raw) {{
  std::fill(raw.begin(), raw.end(), 0.0f);
}}

int main() {{
  auto model = tflite::FlatBufferModel::BuildFromFile("model.tflite");
  if (!model) {{
    std::printf("cannot load model.tflite\\n");
    return 1;
  }}
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter || interpreter->AllocateTensors() != kTfLiteOk) {{
    std::printf("cannot build the interpreter\\n");
    return 1;
  }}

  std::vector<float> raw(WINDOW_SAMPLES * NUM_CHANNELS);
  float mu[NUM_CHANNELS], sigma[NUM_CHANNELS];
  read_window(raw);
  normalize_window(raw.data(), interpreter->typed_input_tensor<float>(0), mu, sigma);
  if (interpreter->Invoke() != kTfLiteOk) {{
    std::printf("Invoke failed\\n");
    return 1;
  }}
  const float *output = interpreter->typed_output_tensor<float>(0);
{report}
  return 0;
}}
"""


def c2000_example(facts: ModelFacts) -> str:
    if facts.is_classifier:
        extra = _class_array(facts) + "\n"
        report = """        {
            int32_t k = spade_model_decide();  /* temperature + abstention of this run; -1 = uncertain */

            sci_puts(k < 0 ? "uncertain" : CLASS_NAMES[k]);
            sci_puts("\\n");
        }"""
    else:
        extra = ""
        report = """        {
            /* The forecast is normalized: y * sigma + mu returns it to physical units. */
            const float *y = spade_model_output();
            const float *mu = spade_model_mu();
            const float *sigma = spade_model_sigma();
            int32_t c;

            for (c = 0; c < SPADE_INPUT_CHANNELS; c++)
                forecast[c] = y[SPADE_OUTPUT_LEN - SPADE_INPUT_CHANNELS + c] * sigma[c] + mu[c];
        }"""
    forecast = "" if facts.is_classifier else "static float forecast[SPADE_INPUT_CHANNELS];  /* last step of the horizon */\n"
    return f"""/*
 * main.c -- LAUNCHXL-F28379D, replacing main.c in the exported kit. Channel order: {_channel_comment(facts)}.
 * The C28x has no 8-bit type and int is 16 bits: keep indices in int32_t.
 */
#include "board.h"
#include "spade_model.h"

{extra}{forecast}/* In the arena region: the default .bss region is too small for a whole window. */
#pragma DATA_SECTION(raw, ".spade_arena")
static float raw[SPADE_INPUT_LEN];

/* TODO: replace with your ADC reads. Physical units, same units as in training. */
static void read_sample(float *sample)
{{
    int32_t c;

    for (c = 0; c < SPADE_INPUT_CHANNELS; c++)
        sample[c] = 0.0f;
}}

void main(void)
{{
    board_init();
    sci_init(115200);

    for (;;) {{
        int32_t t;

        for (t = 0; t < SPADE_INPUT_STEPS; t++) {{
            read_sample(&raw[t * SPADE_INPUT_CHANNELS]);
            DEVICE_DELAY_US(200);  /* 5 kHz, the rate of the training data. Use an ADC trigger for real sampling. */
        }}
        spade_model_run(raw);  /* normalizes the window and runs the model */
{report}
    }}
}}
"""
