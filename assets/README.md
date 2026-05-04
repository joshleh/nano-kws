# Bundled assets

This directory holds **committed** artefacts so that a fresh clone can run
the benchmark and live demo without training:

| File | Produced by | Purpose |
| --- | --- | --- |
| `ds_cnn_small.pt`         | `make train`     | Float checkpoint (also used by `make quantize` / `make export`). |
| `ds_cnn_small_fp32.onnx`  | `make export`    | fp32 ONNX baseline for the benchmark. |
| `ds_cnn_small_int8.onnx`  | `make quantize`  | Static-PTQ INT8 ONNX. **The canonical demo asset.** |
| `label_map.json`          | `make quantize`  | `{index: label}` mapping bundled with the model. |
| `benchmark_table.md`      | `make benchmark` | Rendered Markdown snippet that the README embeds. |
| `arch_diagram.png`        | manual           | Architecture diagram for the README. |

These files are populated as Phases 2–4 land. They are intentionally
small (~few hundred KB total) so they can live in `git` directly.
