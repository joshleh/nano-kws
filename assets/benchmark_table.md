| Variant | Runtime | Params | MACs | Top-1 acc | Size on disk | Latency mean | Latency p50 | Latency p95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DS-CNN small fp32 | PyTorch (CPU) | 62.1 K | 44.13 M | 18.86% | n/a | 5.919 ms | 4.954 ms | 12.500 ms |
| DS-CNN small fp32 | ONNX Runtime (CPU) | 62.1 K | 44.13 M | 18.86% | 242.6 KB | 0.479 ms | 0.466 ms | 0.610 ms |
| DS-CNN small INT8 | ONNX Runtime (CPU) | 62.1 K | 44.13 M | 17.84% | 93.4 KB | 0.378 ms | 0.379 ms | 0.513 ms |

**INT8 vs fp32 (ONNX Runtime):**
- Size: 38.5% of fp32 (242.6 KB -> 93.4 KB)
- Latency: 1.27x (0.479 ms -> 0.378 ms mean)
- Top-1 accuracy: -1.02 pp
