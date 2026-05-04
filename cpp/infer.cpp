// nano-kws — minimal C++ inference harness for the bundled INT8 ONNX model.
//
// Usage:
//   nano_kws_infer <model.onnx> [--iters N] [--warmup N]
//
// The harness intentionally does not depend on libsndfile or any DSP
// library: it generates a synthetic log-mel spectrogram of the right
// shape (1 x 1 x N_MELS x N_FRAMES) and times the *pure model forward
// pass* through ONNX Runtime's C++ API. The point is to demonstrate
// loading the same artefact the Python pipeline ships and to measure
// host-CPU inference latency without the Python interpreter or
// torchaudio in the way.
//
// Build (see cpp/README.md for full instructions):
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
//         -Donnxruntime_ROOT=/path/to/onnxruntime
//   cmake --build build --config Release
//
// Run:
//   ./build/nano_kws_infer ../assets/ds_cnn_small_int8.onnx --iters 1000

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

// Match nano_kws/config.py exactly so this binary consumes the same
// artefact the Python pipeline produces.
constexpr int64_t kNMels = 40;
constexpr int64_t kNFrames = 97;
constexpr int64_t kNumClasses = 12;
constexpr std::array<const char*, kNumClasses> kLabels = {
    "yes",  "no",  "up",  "down", "left",      "right",
    "on",   "off", "stop", "go",  "_silence_", "_unknown_",
};

struct Args {
    std::string model_path;
    int warmup = 50;
    int iters = 500;
};

void PrintUsage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " <model.onnx> [--iters N] [--warmup N]\n";
}

bool ParseArgs(int argc, char** argv, Args& out) {
    if (argc < 2) {
        PrintUsage(argv[0]);
        return false;
    }
    out.model_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string flag = argv[i];
        const bool has_value = (i + 1) < argc;
        if (flag == "--iters" && has_value) {
            out.iters = std::atoi(argv[++i]);
        } else if (flag == "--warmup" && has_value) {
            out.warmup = std::atoi(argv[++i]);
        } else {
            std::cerr << "unknown / incomplete arg: " << flag << "\n";
            PrintUsage(argv[0]);
            return false;
        }
    }
    return true;
}

std::vector<float> MakeSyntheticInput(std::mt19937& rng) {
    // Distribution-plausible log-mel: roughly N(-4, 2), clipped to [-12, 4].
    std::normal_distribution<float> dist(-4.0f, 2.0f);
    std::vector<float> input(kNMels * kNFrames);
    for (auto& x : input) {
        x = std::clamp(dist(rng), -12.0f, 4.0f);
    }
    return input;
}

double Percentile(std::vector<double> xs, double q) {
    if (xs.empty()) return 0.0;
    const auto n = static_cast<size_t>(q * (xs.size() - 1));
    std::nth_element(xs.begin(), xs.begin() + n, xs.end());
    return xs[n];
}

}  // namespace

int main(int argc, char** argv) {
    Args args;
    if (!ParseArgs(argc, argv, args)) return 1;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "nano_kws_infer");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // ORTCHAR_T is wchar_t on Windows, char elsewhere — let std::filesystem
    // pick the right native encoding instead of a naive char->wchar cast.
    const std::filesystem::path model_fs(args.model_path);
    Ort::Session session(env, model_fs.c_str(), opts);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr in_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr out_name = session.GetOutputNameAllocated(0, allocator);
    const std::array<const char*, 1> input_names = {in_name.get()};
    const std::array<const char*, 1> output_names = {out_name.get()};

    const std::array<int64_t, 4> input_shape = {1, 1, kNMels, kNFrames};
    std::mt19937 rng(0xC0FFEE);
    auto input_data = MakeSyntheticInput(rng);

    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    auto run_once = [&]() {
        return session.Run(Ort::RunOptions{nullptr},
                           input_names.data(), &input_tensor, 1,
                           output_names.data(), 1);
    };

    std::cout << "Model: " << args.model_path << "\n";
    std::cout << "Input: " << input_names[0]
              << " [1, 1, " << kNMels << ", " << kNFrames << "]\n";
    std::cout << "Threads: 1 intra-op (single-thread latency)\n";
    std::cout << "Warmup iters: " << args.warmup
              << " | Timed iters: " << args.iters << "\n\n";

    for (int i = 0; i < args.warmup; ++i) (void)run_once();

    std::vector<double> times_ms;
    times_ms.reserve(args.iters);
    using clock = std::chrono::steady_clock;
    for (int i = 0; i < args.iters; ++i) {
        const auto t0 = clock::now();
        auto outputs = run_once();
        const auto t1 = clock::now();
        times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());

        if (i == 0) {  // print top-1 once so stdout proves a real forward.
            const auto* logits = outputs[0].GetTensorMutableData<float>();
            const auto top = std::distance(
                logits, std::max_element(logits, logits + kNumClasses));
            std::cout << "First-iter top-1 (synthetic input): "
                      << kLabels[top] << " (logit=" << logits[top] << ")\n\n";
        }
    }

    const double mean = std::accumulate(times_ms.begin(), times_ms.end(), 0.0) / times_ms.size();
    const double p50 = Percentile(times_ms, 0.50);
    const double p95 = Percentile(times_ms, 0.95);
    const double p99 = Percentile(times_ms, 0.99);

    std::cout << "Latency (ms) — mean " << mean
              << " | p50 " << p50
              << " | p95 " << p95
              << " | p99 " << p99 << "\n";
    return 0;
}
