#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::exit(1);
    }
}

static std::vector<float> evaluate(ggml_backend_t backend, ggml_type type, const void * weights,
                                  const std::vector<float> & hidden, const std::vector<int32_t> & routes,
                                  int width, int outputs, int experts, int rows_per_token, int tokens) {
    constexpr int topk = 8;
    auto * ctx = ggml_init({ggml_tensor_overhead() * 10 + ggml_graph_overhead(), nullptr, true});
    auto * w = ggml_new_tensor_3d(ctx, type, width, outputs, experts);
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, rows_per_token, tokens);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, topk, tokens);
    auto * y = ggml_mul_mat_id(ctx, w, x, ids);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "Cannot allocate test tensors");
    ggml_backend_tensor_set(w, weights, 0, ggml_nbytes(w));
    ggml_backend_tensor_set(x, hidden.data(), 0, ggml_nbytes(x));
    ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "MUL_MAT_ID failed");
    if (type == GGML_TYPE_NVFP4) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i) {
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "MUL_MAT_ID failed");
        }
        const auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start);
        std::printf("%s K=%d N=%d tokens=%d mean_ms=%.3f\n", ggml_backend_name(backend), width, outputs,
                    tokens, elapsed.count() / 10);
    }
    std::vector<float> result(ggml_nelements(y));
    ggml_backend_tensor_get(y, result.data(), 0, ggml_nbytes(y));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

int main() {
    ggml_backend_load_all();
    auto * cpu = ggml_backend_cpu_init();
    auto * xpu = ggml_backend_init_by_name("SYCL0", nullptr);
    require(xpu != nullptr, "SYCL0 is unavailable");
    ggml_backend_cpu_set_n_threads(cpu, 6);
    std::mt19937 rng(837);
    std::normal_distribution<float> normal(0.0f, 0.05f);
    constexpr int experts = 16, outputs = 2304, topk = 8, tokens = 64;
    for (const int width : {2304, 1152}) {
        const int rows_per_token = width == 2304 ? 1 : topk;
        const size_t packed_row = ggml_row_size(GGML_TYPE_NVFP4, width);
        std::vector<float> weights(size_t(experts) * outputs * width);
        std::vector<float> hidden(size_t(tokens) * rows_per_token * width);
        std::vector<int32_t> routes(tokens * topk);
        for (auto & sample : weights) { sample = normal(rng); }
        for (auto & sample : hidden) { sample = normal(rng); }
        for (int token = 0; token < tokens; ++token) {
            for (int rank = 0; rank < topk; ++rank) {
                routes[token * topk + rank] = (token + rank) % experts;
            }
        }
        routes[3] = -1;
        std::vector<uint8_t> packed(packed_row * outputs * experts);
        ggml_quantize_chunk(GGML_TYPE_NVFP4, weights.data(), packed.data(), 0, outputs * experts, width, nullptr);
        auto * dequantize = ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float;
        for (int row = 0; row < outputs * experts; ++row) {
            dequantize(packed.data() + row * packed_row, weights.data() + size_t(row) * width, width);
        }
        const auto expected = evaluate(cpu, GGML_TYPE_F32, weights.data(), hidden, routes,
                                       width, outputs, experts, rows_per_token, tokens);
        const auto actual = evaluate(xpu, GGML_TYPE_NVFP4, packed.data(), hidden, routes,
                                     width, outputs, experts, rows_per_token, tokens);
        double error = 0, norm = 0;
        for (size_t i = 0; i < expected.size(); ++i) {
            error += std::pow(double(expected[i]) - actual[i], 2);
            norm += double(expected[i]) * expected[i];
        }
        const double relative = std::sqrt(error / norm);
        std::printf("relative_L2=%.6f\n", relative);
        require(relative < 0.005, "NVFP4 XPU output deviates from the CPU dequantized reference");
        for (int i = 3 * outputs; i < 4 * outputs; ++i) {
            require(actual[i] == 0, "Skipped expert must leave a zero output row");
        }
    }
    ggml_backend_free(xpu);
    ggml_backend_free(cpu);
}
