# AI-N36: where the B4 ESP32 int8 loses inner_race (quantization diagnosis)

**Simulated.** Decision M-D3 (2026-11-12): **validation split of the B4 3-class training data only; E4 and E4c were not read.** A diagnosis, not a candidate: nothing trained, nothing promoted. Command: `python experiments/ai_n36_quant_diagnosis.py --work /mnt/datos/spade/m4_work` (19 s wall). Raw: [`ai_n36_quant_diagnosis.json`](ai_n36_quant_diagnosis.json).

Model: `esp32_dscnn/2026-11-10_run01` (B4 4a) and its selected int8 variant V1 (per tensor, noise-free stratified calibration). Validation windows: 1796 (534 at 40/60 rad/s, 1222 at 80–220 rad/s, 40 off-grid healthy). Recall and macro-F1 by argmax.

**Method.** Exact float emulation of TFLite post-training quantization:
- BatchNorm folded into the pointwise convolutions;
- weights symmetric per tensor;
- activations asymmetric per tensor at the tensors TFLite quantizes (input, A1 = depthwise 1, A2 = pointwise 1 + ReLU, A3 = depthwise 2, A4 = pointwise 2 + ReLU, A5 = GAP, A6 = logits), with the min/max of V1's own calibration set.

**Checks:** the folded float model agrees with Keras on 1.000 of the windows, and the emulated full int8 with the **real V1 file** on 0.9989.

## 1. Where inner_race is lost

| Configuration | Low 40/60: macro-F1 | inner | normal | outer | Grid 80–220: macro-F1 | inner | All: macro-F1 |
|---|---|---|---|---|---|---|---|
| Keras float | 0.760 | 0.925 | 0.392 | 0.925 | 0.850 | 0.817 | 0.828 |
| weights int8 only | 0.745 | 0.950 | 0.383 | 0.881 | 0.850 | 0.831 | 0.823 |
| input int8 only | 0.758 | 0.944 | 0.400 | 0.901 | 0.843 | 0.838 | 0.822 |
| activations int8 only | 0.844 | 0.665 | 0.925 | 0.937 | 0.869 | 0.789 | 0.861 |
| full int8 (emulated) | 0.876 | 0.814 | 0.908 | 0.913 | 0.864 | 0.792 | 0.871 |
| full int8 (**real V1 file**) | 0.876 | 0.814 | 0.908 | 0.913 | 0.863 | 0.792 | 0.870 |

**One activation tensor at a time** (everything else float):

| Configuration | Low 40/60: macro-F1 | inner | normal | outer | Grid 80–220: macro-F1 | inner | All: macro-F1 |
|---|---|---|---|---|---|---|---|
| only A1_dw1 | 0.877 | 0.870 | 0.883 | 0.870 | 0.864 | 0.801 | 0.867 |
| only A2_pw1_relu | 0.756 | 0.919 | 0.383 | 0.929 | 0.855 | 0.815 | 0.831 |
| only A3_dw2 | 0.652 | 0.596 | 0.325 | 0.960 | 0.806 | 0.792 | 0.762 |
| only A4_pw2_relu | 0.765 | 0.913 | 0.408 | 0.929 | 0.853 | 0.817 | 0.831 |
| only A5_gap | 0.759 | 0.932 | 0.383 | 0.929 | 0.850 | 0.817 | 0.828 |
| only A6_logits | 0.763 | 0.938 | 0.392 | 0.925 | 0.850 | 0.817 | 0.829 |

**Reading.**
- **Weights and input are harmless.** Weights int8 only: inner 0.950 at low speed; input int8 only: 0.944. Float: 0.925.
- **The activations move the result**, mainly two tensors:
  - **A3** (output of the second depthwise conv) alone takes low-speed inner_race from 0.925 to 0.596;
  - **A1** (first depthwise) alone trades inner for normal (inner 0.870, normal 0.883).
- **No activation is clipped:** the share of validation activations outside the calibration range is 0 at every tensor at low speed (≤ 0.05 % in the grid). The loss comes from **rounding resolution**, not from the calibration range.
- **The float model is itself fragile at low speed:** at 40/60 rad/s Keras gets normal recall **0.392** on validation. Full int8 raises normal to 0.908 and lowers inner to 0.814; overall validation macro-F1 is even higher in int8 (0.871 against 0.828).
  - int8 rounding noise at A1/A3 **moves a marginal inner/normal boundary**; it does not erase information.
  - The large E4 loss (`b4_results.md`: inner_race 1.000 → 0.346 at 50/70 rad/s) is the same mechanism on a set this diagnosis may not read.

## 2. Hypotheses (each alone)

| Configuration | Low 40/60: macro-F1 | inner | normal | outer | Grid 80–220: macro-F1 | inner | All: macro-F1 |
|---|---|---|---|---|---|---|---|
| (a) calibration with 40/60 rad/s in the validation share — emulated | 0.875 | 0.814 | 0.900 | 0.917 | 0.864 | 0.792 | 0.870 |
| (a) — **real TFLite** | 0.876 | 0.814 | 0.908 | 0.913 | 0.863 | 0.792 | 0.871 |
| (b) per-channel input scaling before quantizing, folded into depthwise 1 — emulated | 0.548 | 0.031 | 0.892 | 0.937 | 0.831 | 0.703 | 0.772 |
| (b) — **real TFLite** | 0.552 | 0.037 | 0.892 | 0.937 | 0.831 | 0.703 | 0.773 |
| (b) input quantization only | 0.753 | 0.938 | 0.392 | 0.901 | 0.845 | 0.833 | 0.822 |

- **(a) does not recover anything.** inner_race stays at 0.814.
  - The new calibration set has 90/60/138 low-speed windows per class (inner/normal/outer), matching the validation shares of 27 %, 20 % and 42 %.
  - The ranges hardly change, which is consistent with the absence of clipping.
- **(b) makes it much worse:** inner_race 0.03 at low speed.
  - The folding is exact in float (agreement 1.000), and quantizing only the input is harmless (0.938).
  - But the scales differ by about 7×: 21.4 for the accelerations against 3.0 for rpm.
  - So in the folded depthwise kernel the rpm taps become small next to the others. With **one weight scale per tensor**, the rpm information is rounded away.
  - (b) also changes the deployed preprocessing (divide each channel by s_c). **Not recommended.**

## 3. Exploratory (not requested by M-D3; it only informs the next pre-registration)

| Configuration | Low 40/60: macro-F1 | inner | normal | outer | Grid 80–220: macro-F1 | inner | All: macro-F1 |
|---|---|---|---|---|---|---|---|
| activations 16-bit, weights int8 (emulated) | 0.738 | 0.950 | 0.367 | 0.881 | 0.851 | 0.833 | 0.822 |
| only A3 at 16-bit, rest int8 (emulated) | 0.807 | 0.944 | 0.617 | 0.830 | 0.858 | 0.801 | 0.837 |

- **16-bit activations bring back the float behavior:** inner 0.950 at low speed, but normal also returns to 0.367.
- Raising A3 alone to 16 bits gives inner 0.944 and normal 0.617 at low speed.
- **Quantization alone cannot give a model better than its float version.** Here the float version is weak at 40/60 rad/s.

## 4. `esp32_tiny` (M4 step 3): not executed

- `esp32_tiny` is the **vsc_dpc regressor** (256 → 64 forecaster, `configs/regressors/esp32_tiny.yaml`), not a bearing classifier.
- It has no `inner_race`, no B4 validation set and no class-level metric. The step cannot be done as written.
- No smaller ESP32 classifier exists in the repo. `esp32_dscnn` needs 68.5 ms against a 51.2 ms window at 10 kHz (AI-N26).
- A realistic ESP32 classifier would have to be **designed and trained**. That is a new decision (question for Luciano).

## 5. Proposal for the next ESP32 pre-registration (not executed)

1. **Target the float model's margin at low speed first.** The pre-registered gate should be evaluated on the Keras model before any export, with the low-speed recall per class. In B4, normal was 0.392 at 40/60 rad/s on validation.
2. **Architecture within the latency budget:** a smaller or strided DS-CNN must fit 51.2 ms (512 samples) or 25.6 ms (256 samples) on the C2000 reference. Pre-register its latency on the board as a criterion.
3. **Quantization:**
   - per-tensor int8 (V1 recipe) as the baseline;
   - int16 activations / int8 weights (TFLite `EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8`) as an alternative, **only if TFLite Micro on the ESP32 supports those kernels: to verify, not verified here**;
   - quantization-aware training would need `tensorflow-model-optimization`, **a new dependency: proposal only, not added**.
4. **Selection on validation, test on a held-out low-speed set** that no decision has seen (like E4c), with the int8-vs-Keras limits of D-B4-2.
5. **Declared:** this diagnosis looked only at validation, so its conclusions can shape the next design without contaminating E4/E4c.
