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

The committed canonical artefacts come from a **30-epoch CPU run** of
the Phase 5 sweep (`python -m scripts.sweep_sizes`) on Google Speech
Commands v0.02 train split, with the default augmentation stack
(SpecAugment + background noise mixing) and AdamW + cosine LR. The
sweep also produced w=0.25 and w=1.0 variants — those are kept under
`runs/sweep/` (gitignored, regenerable) and summarised by
`assets/sweep_table.md` and `assets/sweep_plot.png`.

Headline numbers (test split, 4888 clips, 12 classes):

- fp32 ONNX top-1: 79.3 %
- INT8 ONNX top-1: 72.2 % (-7.1 pp from PTQ; flagged for QAT follow-up)
- INT8 size: 93.4 KB (38.5 % of fp32)
- INT8 latency: 0.46 ms mean on a single ORT CPU thread

Re-running `make sweep` (or `make train && make quantize && make benchmark`)
overwrites these files in place and the README tables auto-update via
their BEGIN/END markers.
