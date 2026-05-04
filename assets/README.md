# Bundled assets

This directory holds **committed** artefacts so a fresh clone can run the
benchmark and live demo without training. Total footprint is ~600 KB.

| File | Produced by | Purpose |
| --- | --- | --- |
| `ds_cnn_w0p5.pt`                    | `make train`     | PyTorch fp32 checkpoint (DS-CNN, width=0.5, 62K params). Also feeds `make quantize` / `make export`. |
| `ds_cnn_w0p5.history.json`          | `make train`     | Per-epoch loss/accuracy history for plots. |
| `ds_cnn_small_fp32.onnx`            | `make export`    | fp32 ONNX baseline for the benchmark. |
| `ds_cnn_small_fp32.label_map.json`  | `make export`    | `{index: label}` + input-shape sidecar for the fp32 ONNX. |
| `ds_cnn_small_int8.onnx`            | `make quantize`  | Static-PTQ INT8 ONNX. **The canonical demo asset** (~93 KB). |
| `ds_cnn_small_int8.label_map.json`  | `make quantize`  | Sidecar for the INT8 ONNX. |
| `benchmark_table.md`                | `make benchmark` | Rendered Markdown table that the top-level README embeds via `<!-- BEGIN_BENCHMARK_TABLE -->` markers. |

## Provenance of the current snapshot

The committed checkpoint comes from a **1-epoch CPU smoke run** on Google
Speech Commands v0.02, train split, with the default augmentation stack
(SpecAugment + background noise mixing). It exists to prove that the
end-to-end pipeline (`train -> export -> quantize -> benchmark`) works on
a fresh clone, not as a converged model.

Re-running `make train && make quantize && make benchmark` on a real
training budget (30 epochs on a GPU is the planned target) will overwrite
these files in place and refresh the README TL;DR table automatically.
