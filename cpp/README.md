# C++ inference harness (Stretch #2)

A ~150-line reference that loads `assets/ds_cnn_small_int8.onnx` via the
ONNX Runtime C/C++ API and runs single-clip inference outside of any
Python overhead. The point is to demonstrate end-to-end edge inference
in a systems language and to measure host-CPU latency without the
Python interpreter in the way.

Planned layout:

```
cpp/
├── CMakeLists.txt    # find_package(onnxruntime); link against the C API
├── infer.cpp         # load model + label map, read a .wav, print top-1
└── README.md         # this file
```

Build (target):

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
./cpp/build/nano_kws_infer assets/ds_cnn_small_int8.onnx some_clip.wav
```

Implemented in Stretch Phase 6.
