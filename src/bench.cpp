#include "parakeet/parakeet.hpp"

#include <axiom/io/safetensors.hpp>
#include <axiom/system.hpp>
#include <benchmark/benchmark.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ─── Custom CLI flags ───────────────────────────────────────────────────────

static std::string flag_110m;
static std::string flag_tdt_600m;
static std::string flag_rnnt_600m;
static std::string flag_sortformer;
static bool flag_no_gpu = false;
static bool flag_markdown = false;

static void parse_custom_flags(int *argc, char **argv) {
    int out = 1;
    for (int i = 1; i < *argc; ++i) {
        std::string arg = argv[i];
        if (arg.starts_with("--110m="))
            flag_110m = arg.substr(7);
        else if (arg.starts_with("--tdt-600m="))
            flag_tdt_600m = arg.substr(11);
        else if (arg.starts_with("--rnnt-600m="))
            flag_rnnt_600m = arg.substr(12);
        else if (arg.starts_with("--sortformer="))
            flag_sortformer = arg.substr(13);
        else if (arg == "--no-gpu")
            flag_no_gpu = true;
        else if (arg == "--markdown")
            flag_markdown = true;
        else
            argv[out++] = argv[i];
    }
    *argc = out;
}

// ─── Markdown reporter ──────────────────────────────────────────────────────

// Parse audio_sec from benchmark name like "110m_CPU/5/real_time"
static int parse_audio_sec(const std::string &name) {
    // Find the Arg value between first and second '/'
    auto first_slash = name.find('/');
    if (first_slash == std::string::npos)
        return 0;
    auto second_slash = name.find('/', first_slash + 1);
    std::string arg_str = (second_slash != std::string::npos)
                              ? name.substr(first_slash + 1,
                                            second_slash - first_slash - 1)
                              : name.substr(first_slash + 1);
    try {
        return std::stoi(arg_str);
    } catch (...) {
        return 0;
    }
}

// Parse model and device from "110m_CPU/5/real_time" → ("110m", "CPU")
static std::pair<std::string, std::string>
parse_model_device(const std::string &name) {
    auto slash = name.find('/');
    std::string prefix = (slash != std::string::npos) ? name.substr(0, slash) : name;
    auto underscore = prefix.rfind('_');
    if (underscore != std::string::npos)
        return {prefix.substr(0, underscore), prefix.substr(underscore + 1)};
    return {prefix, ""};
}

class MarkdownReporter : public benchmark::BenchmarkReporter {
  public:
    bool ReportContext(const Context &) override {
        std::cerr << "Running benchmarks..." << std::endl;
        return true;
    }

    void ReportRuns(const std::vector<Run> &reports) override {
        for (const auto &r : reports)
            runs_.push_back(r);
    }

    void Finalize() override {
        if (runs_.empty())
            return;

        std::cout
            << "| Model | Device | Audio (s) | Time (ms) | RTF | Throughput |\n";
        std::cout
            << "|-------|--------|-----------|-----------|-----|------------|\n";

        for (const auto &r : runs_) {
            if (r.skipped != benchmark::internal::NotSkipped)
                continue;

            auto [model, device] = parse_model_device(r.benchmark_name());
            int audio_sec = parse_audio_sec(r.benchmark_name());
            double time_ms =
                r.real_accumulated_time / static_cast<double>(r.iterations) *
                1000.0;
            double rtf =
                audio_sec > 0 ? (time_ms / 1000.0) / audio_sec : 0;
            double throughput = rtf > 0 ? 1.0 / rtf : 0;

            std::cout << "| " << model << " | " << device << " | "
                      << audio_sec << " | " << std::fixed
                      << std::setprecision(1) << time_ms << " | "
                      << std::setprecision(4) << rtf << " | "
                      << std::setprecision(0) << throughput << "x |\n";
        }
    }

  private:
    std::vector<Run> runs_;
};

// ─── Model cache ────────────────────────────────────────────────────────────

template <typename Model> struct ModelCache {
    std::unique_ptr<Model> cpu;
    std::unique_ptr<Model> gpu;
};

static ModelCache<parakeet::ParakeetTDTCTC> cache_110m;
static ModelCache<parakeet::ParakeetTDT> cache_tdt_600m;
static ModelCache<parakeet::ParakeetRNNT> cache_rnnt_600m;
static ModelCache<parakeet::Sortformer> cache_sortformer;

template <typename Model, typename Config>
Model &load_model(ModelCache<Model> &cache, const std::string &path,
                  Config config, bool gpu) {
    auto &slot = gpu ? cache.gpu : cache.cpu;
    if (!slot) {
        slot = std::make_unique<Model>(config);
        auto weights = axiom::io::safetensors::load(path);
        slot->load_state_dict(weights, "", false);
        if (gpu)
            slot->to(axiom::Device::GPU);
        std::cerr << "Loaded " << path << (gpu ? " (GPU)" : " (CPU)")
                  << std::endl;
    }
    return *slot;
}

// ─── Benchmark registration ─────────────────────────────────────────────────

static const std::vector<int64_t> audio_durations = {1, 5, 10, 30, 60};

static void add_duration_args(benchmark::Benchmark *b) {
    for (auto d : audio_durations)
        b->Arg(d);
    b->UseRealTime()->Unit(benchmark::kMillisecond);
}

// Force tensor materialization (ensures lazy GPU graphs actually execute).
// Calling strides() triggers materialize_if_needed() inside axiom, which
// compiles and runs the GPU graph without copying data to CPU.
static void materialize(const axiom::Tensor &t) {
    auto s = t.strides();
    benchmark::DoNotOptimize(s);
}

static void register_benchmarks() {
    bool has_gpu = !flag_no_gpu && axiom::system::is_metal_available();

    // 110M TDT-CTC encoder
    if (!flag_110m.empty()) {
        auto cfg = parakeet::make_110m_config();
        int mel = cfg.encoder.mel_bins;

        auto reg = [mel](bool gpu) {
            std::string name =
                std::string("110m_") + (gpu ? "GPU" : "CPU");
            add_duration_args(benchmark::RegisterBenchmark(
                name, [gpu, mel](benchmark::State &state) {
                    auto &model = load_model(cache_110m, flag_110m,
                                             parakeet::make_110m_config(), gpu);
                    int audio_sec = static_cast<int>(state.range(0));
                    auto n_frames = static_cast<size_t>(audio_sec * 100);
                    auto features = axiom::Tensor::randn(
                        {1, n_frames, static_cast<size_t>(mel)});
                    if (gpu)
                        features = features.gpu();

                    // Warmup: compile GPU graph outside timed loop
                    materialize(model.encoder()(features));

                    for (auto _ : state) {
                        auto out = model.encoder()(features);
                        materialize(out);
                    }

                    state.counters["Throughput"] =
                        benchmark::Counter(audio_sec, benchmark::Counter::kIsRate);
                }));
        };

        reg(false);
        if (has_gpu)
            reg(true);
    }

    // TDT 600M encoder
    if (!flag_tdt_600m.empty()) {
        auto cfg = parakeet::make_tdt_600m_config();
        int mel = cfg.encoder.mel_bins;

        auto reg = [mel](bool gpu) {
            std::string name =
                std::string("tdt-600m_") + (gpu ? "GPU" : "CPU");
            add_duration_args(benchmark::RegisterBenchmark(
                name, [gpu, mel](benchmark::State &state) {
                    auto &model =
                        load_model(cache_tdt_600m, flag_tdt_600m,
                                   parakeet::make_tdt_600m_config(), gpu);
                    int audio_sec = static_cast<int>(state.range(0));
                    auto n_frames = static_cast<size_t>(audio_sec * 100);
                    auto features = axiom::Tensor::randn(
                        {1, n_frames, static_cast<size_t>(mel)});
                    if (gpu)
                        features = features.gpu();

                    materialize(model.encoder()(features));

                    for (auto _ : state) {
                        auto out = model.encoder()(features);
                        materialize(out);
                    }

                    state.counters["Throughput"] =
                        benchmark::Counter(audio_sec, benchmark::Counter::kIsRate);
                }));
        };

        reg(false);
        if (has_gpu)
            reg(true);
    }

    // RNNT 600M encoder
    if (!flag_rnnt_600m.empty()) {
        auto cfg = parakeet::make_rnnt_600m_config();
        int mel = cfg.encoder.mel_bins;

        auto reg = [mel](bool gpu) {
            std::string name =
                std::string("rnnt-600m_") + (gpu ? "GPU" : "CPU");
            add_duration_args(benchmark::RegisterBenchmark(
                name, [gpu, mel](benchmark::State &state) {
                    auto &model =
                        load_model(cache_rnnt_600m, flag_rnnt_600m,
                                   parakeet::make_rnnt_600m_config(), gpu);
                    int audio_sec = static_cast<int>(state.range(0));
                    auto n_frames = static_cast<size_t>(audio_sec * 100);
                    auto features = axiom::Tensor::randn(
                        {1, n_frames, static_cast<size_t>(mel)});
                    if (gpu)
                        features = features.gpu();

                    materialize(model.encoder()(features));

                    for (auto _ : state) {
                        auto out = model.encoder()(features);
                        materialize(out);
                    }

                    state.counters["Throughput"] =
                        benchmark::Counter(audio_sec, benchmark::Counter::kIsRate);
                }));
        };

        reg(false);
        if (has_gpu)
            reg(true);
    }

    // Sortformer forward pass
    if (!flag_sortformer.empty()) {
        auto cfg = parakeet::make_sortformer_117m_config();
        int mel = cfg.nest_encoder.mel_bins;

        auto reg = [mel](bool gpu) {
            std::string name =
                std::string("sortformer_") + (gpu ? "GPU" : "CPU");
            add_duration_args(benchmark::RegisterBenchmark(
                name, [gpu, mel](benchmark::State &state) {
                    auto &model = load_model(
                        cache_sortformer, flag_sortformer,
                        parakeet::make_sortformer_117m_config(), gpu);
                    int audio_sec = static_cast<int>(state.range(0));
                    auto n_frames = static_cast<size_t>(audio_sec * 100);
                    auto features = axiom::Tensor::randn(
                        {1, n_frames, static_cast<size_t>(mel)});
                    if (gpu)
                        features = features.gpu();

                    materialize(model.forward(features));

                    for (auto _ : state) {
                        auto out = model.forward(features);
                        materialize(out);
                    }

                    state.counters["Throughput"] =
                        benchmark::Counter(audio_sec, benchmark::Counter::kIsRate);
                }));
        };

        reg(false);
        if (has_gpu)
            reg(true);
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    parse_custom_flags(&argc, argv);

    if (flag_110m.empty() && flag_tdt_600m.empty() && flag_rnnt_600m.empty() &&
        flag_sortformer.empty()) {
        std::cerr
            << "Usage: parakeet_bench [options] [benchmark flags]\n\n"
            << "Model flags (at least one required):\n"
            << "  --110m=PATH         110M TDT-CTC model weights\n"
            << "  --tdt-600m=PATH     600M TDT model weights\n"
            << "  --rnnt-600m=PATH    600M RNNT model weights\n"
            << "  --sortformer=PATH   Sortformer model weights\n"
            << "\nOptions:\n"
            << "  --no-gpu            Skip GPU benchmarks\n"
            << "  --markdown          Output as markdown table\n"
            << "\nGoogle Benchmark flags (passed through):\n"
            << "  --benchmark_filter=REGEX\n"
            << "  --benchmark_repetitions=N\n"
            << "  --benchmark_format={console|json|csv}\n"
            << std::endl;
        return 1;
    }

    benchmark::Initialize(&argc, argv);
    register_benchmarks();

    if (flag_markdown) {
        MarkdownReporter reporter;
        benchmark::RunSpecifiedBenchmarks(&reporter);
    } else {
        benchmark::RunSpecifiedBenchmarks();
    }

    benchmark::Shutdown();
    return 0;
}
