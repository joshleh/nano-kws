Inputs: `(C, H, W) = (56, 16, 47)`, fp32, single-thread (`torch.set_num_threads(1)`). All times wall-clock from `perf_counter_ns`.

**Legend.** *Speedup vs C naive* divides each row's mean wall-time by the hand-written C scalar (`C naive`) baseline, so it answers "how much faster than a textbook nested-loop C kernel?". *Correct?* compares the implementation's output to ATen element-wise (max absolute error `<= --atol`, default 1e-3); `ref` marks the ATen reference itself.

> **Note.** The hand-written C kernels were not built on this machine, so the *C naive* and *C AVX2* rows are absent and the *Speedup vs C naive* column is blank (no baseline to divide by). Run `make microbench-build && make microbench` on a host with CMake + a C compiler with AVX2 to populate them.

### Pointwise (1x1)

| Implementation | Mean (ms) | p50 (ms) | p95 (ms) | Speedup vs C naive | Correct? |
| --- | ---: | ---: | ---: | ---: | :---: |
| ATen (reference) | 0.1150 | 0.0897 | 0.2011 | n/a | ref |
| NumPy einsum | 0.4033 | 0.3523 | 0.6034 | n/a | yes (err 7.6e-06) |

### Depthwise 3x3

| Implementation | Mean (ms) | p50 (ms) | p95 (ms) | Speedup vs C naive | Correct? |
| --- | ---: | ---: | ---: | ---: | :---: |
| ATen (reference) | 0.1936 | 0.1905 | 0.2148 | n/a | ref |
