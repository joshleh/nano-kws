| Variant | Runtime | Params | MACs | Top-1 acc | Size on disk | Latency mean | Latency p50 | Latency p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-CNN small fp32 | PyTorch (CPU) | 62.1 K | 44.13 M | 79.32% | n/a | 3.866 ms | 3.758 ms | 5.652 ms |
| DS-CNN small fp32 | ONNX Runtime (CPU) | 62.1 K | 44.13 M | 79.32% | 242.6 KB | 0.586 ms | 0.496 ms | 0.945 ms |
| DS-CNN small INT8 (PTQ) | ONNX Runtime (CPU) | 62.1 K | 44.13 M | 72.20% | 93.4 KB | 0.413 ms | 0.405 ms | 0.526 ms |
| DS-CNN small INT8 (QAT) | ONNX Runtime (CPU) | 62.1 K | 44.13 M | 90.67% | 93.4 KB | 0.436 ms | 0.424 ms | 0.514 ms |

**INT8 (PTQ) vs fp32 (ONNX Runtime):**
- Size: 38.5% of fp32 (242.6 KB -> 93.4 KB)
- Latency: 1.42x (0.586 ms -> 0.413 ms mean)
- Top-1 accuracy: -7.12 pp

**INT8 (QAT) vs fp32 (ONNX Runtime):**
- Top-1 accuracy: +11.35 pp
- vs PTQ-only INT8: +18.47 pp top-1 recovered by QAT.
