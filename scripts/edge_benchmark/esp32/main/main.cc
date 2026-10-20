// ESP32 TFLite Micro benchmark skeleton -- TODO_CLAUDE_CODE.md P3.
// NOT EXECUTED ON HARDWARE and NOT COMPILED in this repository: it is a starting point for an
// ESP-IDF project with the esp-tflite-micro component. Generate model_data.cc and test_vectors.h
// first (../to_c_array.py). kArenaSize is a guess to be tuned on the device
// (interpreter.arena_used_bytes() reports the real need).
#include <cmath>
#include <cstdio>

#include "esp_timer.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"
#include "test_vectors.h"

namespace {
constexpr int kArenaSize = 64 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];
}  // namespace

extern "C" void app_main(void) {
  const tflite::Model* model = tflite::GetModel(g_model);
  // Builtin ops of the two shipped ESP32 exports, read with driveflow.ai.deployment_status.tflite_op_names:
  // esp32_dscnn/2026-10-18_run01 (classifier) and esp32_tiny/2026-10-19_run01 (forecaster).
  static tflite::MicroMutableOpResolver<15> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddExpandDims();
  resolver.AddFullyConnected();
  resolver.AddMean();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddAdd();
  resolver.AddBatchToSpaceNd();
  resolver.AddMul();
  resolver.AddPack();
  resolver.AddPad();
  resolver.AddShape();
  resolver.AddSpaceToBatchNd();
  resolver.AddStridedSlice();
  static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("AllocateTensors failed -- increase kArenaSize\n");
    return;
  }
  printf("arena used: %u bytes\n", (unsigned)interpreter.arena_used_bytes());
  TfLiteTensor* input = interpreter.input(0);
  TfLiteTensor* output = interpreter.output(0);
  int64_t total_us = 0, worst_us = 0;
  double sq_err = 0.0;
  int top1_agree = 0;
  for (int v = 0; v < kNumVectors; ++v) {
    for (int i = 0; i < kInputSize; ++i) input->data.int8[i] = kInputs[v][i];
    int64_t t0 = esp_timer_get_time();
    if (interpreter.Invoke() != kTfLiteOk) { printf("Invoke failed at vector %d\n", v); return; }
    int64_t dt = esp_timer_get_time() - t0;
    total_us += dt;
    if (dt > worst_us) worst_us = dt;
    int best = 0, best_ref = 0;
    for (int o = 0; o < kOutputSize; ++o) {
      float y = (output->data.int8[o] - kOutputZeroPoint) * kOutputScale;
      sq_err += (y - kExpected[v][o]) * (y - kExpected[v][o]);
      if (y > (output->data.int8[best] - kOutputZeroPoint) * kOutputScale) best = o;
      if (kExpected[v][o] > kExpected[v][best_ref]) best_ref = o;
    }
    top1_agree += (best == best_ref);
  }
  printf("latency mean %lld us, worst %lld us over %d vectors\n", total_us / kNumVectors, worst_us, kNumVectors);
  printf("RMSE vs host TFLite %.6f; top-1 agreement %d/%d (classifiers only)\n", std::sqrt(sq_err / (kNumVectors * kOutputSize)), top1_agree, kNumVectors);
}
