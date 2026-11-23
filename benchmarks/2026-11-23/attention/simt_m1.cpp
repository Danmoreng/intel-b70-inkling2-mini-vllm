// SPDX-License-Identifier: Apache-2.0
// Adapted from SergiioB cookbook af03fa9 for the EXL3 FP16 query, relative
// bias, sink and natural-log merge.
#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <cmath>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <torch/library.h>

namespace {

static inline float bf16_to_f32(uint16_t b) {
  return sycl::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}

static inline uint16_t f32_to_bf16(float f) {
  uint32_t u = sycl::bit_cast<uint32_t>(f);
  uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

static inline float e4m3_to_f32_init(uint8_t u) {
  uint32_t s = static_cast<uint32_t>(u & 0x80u) << 24;
  uint32_t m = static_cast<uint32_t>(u & 0x7Fu);
  if (m == 127u)
    return sycl::bit_cast<float>(s | 0x7fc00000u);
  if (m >= 8u)
    return sycl::bit_cast<float>(s | ((m << 20) + (120u << 23)));
  float v = static_cast<float>(m) * (1.0f / 512.0f);
  return (u & 0x80u) ? -v : v;
}

static inline uint16_t e4m3_to_bf16_init(uint8_t u) {
  return f32_to_bf16(e4m3_to_f32_init(u));
}

static inline float fast_sigmoid(float x) {
  return 1.0f / (1.0f + sycl::native::exp2(-x * 1.4426950408889634f));
}

template <bool IS_FP8>
static inline void load_kv_vec8_f32(const void *base_ptr, int64_t elem_off,
                                    const float *lut, float dst[8]) {
  if constexpr (IS_FP8) {
    const uint8_t *p = static_cast<const uint8_t *>(base_ptr) + elem_off;
    uint64_t packed = *reinterpret_cast<const uint64_t *>(p);
#pragma unroll
    for (int i = 0; i < 8; ++i)
      dst[i] = lut[(packed >> (i * 8)) & 0xFFu];
  } else {
    const uint16_t *p = static_cast<const uint16_t *>(base_ptr) + elem_off;
    sycl::uint4 packed = *reinterpret_cast<const sycl::uint4 *>(p);
    uint32_t w[4] = {packed.x(), packed.y(), packed.z(), packed.w()};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      dst[2 * i] = bf16_to_f32(static_cast<uint16_t>(w[i] & 0xFFFFu));
      dst[2 * i + 1] = bf16_to_f32(static_cast<uint16_t>(w[i] >> 16));
    }
  }
}

template <bool IS_FP8, int G> struct SingleTokSplitKernel {
  const uint16_t *Q;
  const void *K;
  const void *V;
  const float *Rel;
  const int32_t *BlockTable;
  const int32_t *SeqLens;
  const float *Sink;
  float *OPart;
  float *LsePart;

  float softmax_scale;
  int64_t stride_q_t, stride_q_h;
  int64_t stride_kv_blk, stride_kv_slot, stride_kv_h;
  int64_t stride_rel_t, stride_bt;
  int H, E, E_PAD, PAGE, WINDOW, num_tokens, num_splits;

  sycl::local_accessor<float, 1> smem;    // [4 * 3 * 130]
  sycl::local_accessor<float, 1> lut_slm; // [256]

  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<3> it) const {
    constexpr int NUM_SG = 4;
    constexpr int D = 128;
    constexpr float LOG2E = 1.4426950408889634f;

    const int tid = static_cast<int>(it.get_local_linear_id());
    if constexpr (IS_FP8) {
      for (int i = tid; i < 256; i += 64)
        lut_slm[i] = e4m3_to_f32_init(static_cast<uint8_t>(i));
      sycl::group_barrier(it.get_group());
    }
    const float *lut =
        lut_slm.get_multi_ptr<sycl::access::decorated::no>().get();

    const int split = static_cast<int>(it.get_group(0));
    const int kv_head = static_cast<int>(it.get_group(1));
    const int req = static_cast<int>(it.get_group(2));

    const int q_start = 0;
    const int q_len = 1;
    if (q_len <= 0)
      return;
    const int token = q_start;
    const int seqlen_k = SeqLens[req];
    const int past = seqlen_k - q_len;
    const int pos = past;

    const int sg_id = tid >> 4;
    const int lane = tid & 15;
    const int d_base = lane * 8;
    auto sg = it.get_sub_group();

    const int h0 = kv_head * G;
    const float qk_scale = softmax_scale * LOG2E;
    float q_reg[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      const uint16_t *qp =
          Q + token * stride_q_t + (h0 + g) * stride_q_h + d_base;
      sycl::uint4 packed = *reinterpret_cast<const sycl::uint4 *>(qp);
      uint32_t w[4] = {packed.x(), packed.y(), packed.z(), packed.w()};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        q_reg[g][2 * i] = static_cast<float>(sycl::bit_cast<sycl::half>(
                              static_cast<uint16_t>(w[i] & 0xFFFFu))) *
                          qk_scale;
        q_reg[g][2 * i + 1] = static_cast<float>(sycl::bit_cast<sycl::half>(
                                  static_cast<uint16_t>(w[i] >> 16))) *
                              qk_scale;
      }
    }

    const int tile_end = (128 < q_len) ? 128 : q_len;
    const int rel_base = ((past + tile_end + 127) / 128) * 128 - E_PAD;
    const int64_t rel_t_off = static_cast<int64_t>(token) * stride_rel_t;

    const int k_min_all =
        (WINDOW > 0 && pos - (WINDOW - 1) > 0) ? (pos - (WINDOW - 1)) : 0;
    const int k_max_all = (pos + 1 < seqlen_k) ? (pos + 1) : seqlen_k;
    const int total_k = k_max_all - k_min_all;
    const int total_chunks = num_splits * NUM_SG;
    const int my_id = split * NUM_SG + sg_id;
    const int k_per = (total_k + total_chunks - 1) / total_chunks;
    const int my_lo = k_min_all + my_id * k_per;
    const int my_hi = (my_lo + k_per < k_max_all) ? (my_lo + k_per) : k_max_all;

    const int32_t *bt = BlockTable + static_cast<int64_t>(req) * stride_bt;
    const int64_t kv_h_off =
        static_cast<int64_t>(kv_head) * stride_kv_h + d_base;

    float m_i[G], l_i[G], acc[G][8];
#pragma unroll
    for (int g = 0; g < G; ++g) {
      m_i[g] = (split == 0 && sg_id == 0) ? Sink[h0 + g] * LOG2E : -INFINITY;
      l_i[g] = (split == 0 && sg_id == 0) ? 1.0f : 0.0f;
#pragma unroll
      for (int i = 0; i < 8; ++i)
        acc[g][i] = 0.0f;
    }

    int k_cur = my_lo;
    while (k_cur < my_hi) {
      const int p_idx = k_cur / PAGE;
      const int phys_page = bt[p_idx];
      const int page_end = (p_idx + 1) * PAGE;
      const int k_end = (page_end < my_hi) ? page_end : my_hi;
      const int64_t blk_off =
          static_cast<int64_t>(phys_page) * stride_kv_blk + kv_h_off;
      const bool page_has_rel = (pos - k_cur) < (E + PAGE);

      while (k_cur + 4 <= k_end) {
        float kv_k[4][8], kv_v[4][8];
        int slot0 = k_cur - p_idx * PAGE;
#pragma unroll
        for (int u = 0; u < 4; ++u) {
          int64_t off =
              blk_off + static_cast<int64_t>(slot0 + u) * stride_kv_slot;
          load_kv_vec8_f32<IS_FP8>(K, off, lut, kv_k[u]);
          load_kv_vec8_f32<IS_FP8>(V, off, lut, kv_v[u]);
        }
#pragma unroll
        for (int g = 0; g < G; ++g) {
          float s[4];
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i)
              dot += q_reg[g][i] * kv_k[u][i];
            dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
            if (page_has_rel && (pos - (k_cur + u) < E)) {
              dot += Rel[(h0 + g) * E + pos - (k_cur + u)] * LOG2E;
            }
            s[u] = dot;
          }
          float m4 = sycl::fmax(sycl::fmax(s[0], s[1]), sycl::fmax(s[2], s[3]));
          float m_new = sycl::fmax(m_i[g], m4);
          float alpha =
              (m_i[g] == -INFINITY) ? 0.0f : sycl::native::exp2(m_i[g] - m_new);
          float p0 = sycl::native::exp2(s[0] - m_new);
          float p1 = sycl::native::exp2(s[1] - m_new);
          float p2 = sycl::native::exp2(s[2] - m_new);
          float p3 = sycl::native::exp2(s[3] - m_new);
          l_i[g] = l_i[g] * alpha + (p0 + p1 + p2 + p3);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            acc[g][i] = acc[g][i] * alpha + p0 * kv_v[0][i] + p1 * kv_v[1][i] +
                        p2 * kv_v[2][i] + p3 * kv_v[3][i];
          }
          m_i[g] = m_new;
        }
        k_cur += 4;
      }
      while (k_cur < k_end) {
        int slot = k_cur - p_idx * PAGE;
        int64_t off = blk_off + static_cast<int64_t>(slot) * stride_kv_slot;
        float k_vec[8], v_vec[8];
        load_kv_vec8_f32<IS_FP8>(K, off, lut, k_vec);
        load_kv_vec8_f32<IS_FP8>(V, off, lut, v_vec);
#pragma unroll
        for (int g = 0; g < G; ++g) {
          float dot = 0.0f;
#pragma unroll
          for (int i = 0; i < 8; ++i)
            dot += q_reg[g][i] * k_vec[i];
          dot = sycl::reduce_over_group(sg, dot, sycl::plus<float>());
          if (page_has_rel && (pos - k_cur < E)) {
            dot += Rel[(h0 + g) * E + pos - k_cur] * LOG2E;
          }
          float m_new = sycl::fmax(m_i[g], dot);
          float alpha =
              (m_i[g] == -INFINITY) ? 0.0f : sycl::native::exp2(m_i[g] - m_new);
          float p = sycl::native::exp2(dot - m_new);
          l_i[g] = l_i[g] * alpha + p;
#pragma unroll
          for (int i = 0; i < 8; ++i)
            acc[g][i] = acc[g][i] * alpha + p * v_vec[i];
          m_i[g] = m_new;
        }
        k_cur++;
      }
    }

#pragma unroll
    for (int g = 0; g < G; ++g) {
      int base = (g * NUM_SG + sg_id) * 130;
      if (lane == 0) {
        smem[base + 128] = m_i[g];
        smem[base + 129] = l_i[g];
      }
#pragma unroll
      for (int i = 0; i < 8; ++i)
        smem[base + d_base + i] = acc[g][i];
    }
    sycl::group_barrier(it.get_group());

#pragma unroll
    for (int g = 0; g < G; ++g) {
      const int64_t part =
          (static_cast<int64_t>(split) * num_tokens + token) * H + (h0 + g);
      float m_max = -INFINITY;
#pragma unroll
      for (int s = 0; s < NUM_SG; ++s)
        m_max = sycl::fmax(m_max, smem[(g * NUM_SG + s) * 130 + 128]);
      float l_sum = 0.0f, w_s[NUM_SG];
#pragma unroll
      for (int s = 0; s < NUM_SG; ++s) {
        float ls = smem[(g * NUM_SG + s) * 130 + 129];
        w_s[s] =
            (m_max != -INFINITY && ls > 0.0f)
                ? sycl::native::exp2(smem[(g * NUM_SG + s) * 130 + 128] - m_max)
                : 0.0f;
        l_sum += ls * w_s[s];
      }
      if (tid == 0)
        LsePart[part] = (l_sum > 0.0f)
                            ? (m_max + sycl::native::log2(l_sum)) / LOG2E
                            : -INFINITY;
      float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
#pragma unroll
      for (int k2 = 0; k2 < 2; ++k2) {
        int d = tid + k2 * 64;
        float ov = 0.0f;
#pragma unroll
        for (int s = 0; s < NUM_SG; ++s)
          ov += smem[(g * NUM_SG + s) * 130 + d] * w_s[s];
        OPart[part * D + d] = ov * inv_l;
      }
    }
  }
};

void m1(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
        const at::Tensor &bt, const at::Tensor &len, const at::Tensor &rel,
        const at::Tensor &sink, at::Tensor part, at::Tensor lse) {
  TORCH_CHECK(q.is_xpu() && q.sizes() == at::IntArrayRef({1, 24, 128}) &&
              q.is_contiguous() && q.scalar_type() == at::kHalf);
  for (const auto &tensor : {k, v, bt, len, rel, sink, part, lse})
    TORCH_CHECK(tensor.device() == q.device());
  TORCH_CHECK(k.dim() == 4 && k.size(1) > 0 && k.size(2) == 8 &&
              k.size(3) == 128 && k.stride(3) == 1 &&
              k.scalar_type() == at::kFloat8_e4m3fn);
  TORCH_CHECK(v.sizes() == k.sizes() && v.strides() == k.strides() &&
              v.scalar_type() == k.scalar_type());
  TORCH_CHECK(reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(k.data_ptr()) % 8 == 0 &&
              reinterpret_cast<uintptr_t>(v.data_ptr()) % 8 == 0 &&
              k.stride(0) % 8 == 0 && k.stride(1) % 8 == 0 && k.stride(2) % 8 == 0);
  TORCH_CHECK(bt.dim() == 2 && bt.size(0) == 1 && bt.is_contiguous() &&
              bt.scalar_type() == at::kInt);
  TORCH_CHECK(len.numel() == 1 && len.scalar_type() == at::kInt);
  TORCH_CHECK(rel.dim() == 3 && rel.size(0) == 1 && rel.size(1) == 24 &&
              rel.size(2) > 0 && rel.scalar_type() == at::kFloat &&
              rel.is_contiguous());
  TORCH_CHECK(sink.numel() == 24 && sink.scalar_type() == at::kFloat &&
              sink.is_contiguous());
  const int splits = part.size(0);
  TORCH_CHECK(
      splits > 0 && part.sizes() == at::IntArrayRef({splits, 1, 24, 128}) &&
      lse.sizes() == at::IntArrayRef({splits, 1, 24}) &&
      part.scalar_type() == at::kFloat && lse.scalar_type() == at::kFloat &&
      part.is_contiguous() && lse.is_contiguous());
  auto &queue = c10::xpu::getCurrentXPUStream(q.device().index()).queue();
  queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(4 * 3 * 130), cgh),
        lut(sycl::range<1>(256), cgh);
    SingleTokSplitKernel<true, 3> kernel{
        reinterpret_cast<const uint16_t *>(q.data_ptr()),
        k.data_ptr(),
        v.data_ptr(),
        rel.data_ptr<float>(),
        bt.data_ptr<int32_t>(),
        len.data_ptr<int32_t>(),
        sink.data_ptr<float>(),
        part.data_ptr<float>(),
        lse.data_ptr<float>(),
        0.08838834764831844f,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        rel.stride(0),
        bt.stride(0),
        24,
        int(rel.size(2)),
        int(rel.size(2)),
        int(k.size(1)),
        0,
        1,
        splits,
        scratch,
        lut};
    cgh.parallel_for(sycl::nd_range<3>({size_t(splits * 64), 8, 1}, {64, 1, 1}),
                     kernel);
  });
}
} // namespace
TORCH_LIBRARY(inkling_hills, m) {
  m.def("m1(Tensor q, Tensor k, Tensor v, Tensor bt, Tensor len, Tensor rel, "
        "Tensor sink, Tensor(a!) part, Tensor(b!) lse) -> ()");
}
TORCH_LIBRARY_IMPL(inkling_hills, XPU, m) { m.impl("m1", m1); }
