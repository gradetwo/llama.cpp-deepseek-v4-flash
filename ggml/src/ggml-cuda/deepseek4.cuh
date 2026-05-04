#pragma once

#include "common.cuh"

struct rope_corr_dims {
    float v[2];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

template<bool forward>
static __device__ void rope_yarn(
        const float theta_extrap, const float freq_scale, const rope_corr_dims corr_dims,
        const int64_t i0, const float ext_factor,
        float mscale, float & cos_theta, float & sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    cos_theta = cosf(theta) * mscale;
    sin_theta = sinf(theta) * mscale;
    if (!forward) {
        sin_theta *= -1.0f;
    }
}

static __device__ float dsv4_e4m3fn_dequant(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    float ax = fminf(fabsf(x), 448.0f);
    if (ax == 0.0f) return 0.0f;

    int ix = __float_as_int(ax);
    int e = ((ix >> 23) & 0xff) - 127; // actual exponent

    int ef, mant3;

    if (e < -6) {
        // Subnormal range: 0, 2^-9, 2*2^-9, ..., 7*2^-9
        float m = ax * 512.0f; // ax / 2^-9
        mant3 = __float2int_rn(m);
        if (mant3 > 7) mant3 = 7;
        ef = 0;
    } else {
        if (e > 8) e = 8;
        ef = e + 7;
        float scale = ldexpf(1.0f, e);
        float mant_raw = ax / scale - 1.0f;
        mant3 = __float2int_rn(mant_raw * 8.0f);
        if (mant3 >= 8) { mant3 = 0; ef++; }
        if (ef > 15) { ef = 15; mant3 = 7; }
        if (mant3 < 0) mant3 = 0;
    }

    float val = (ef == 0)
        ? (float)mant3 * 0.001953125f          // mant * 2^-9
        : ldexpf(1.0f + (float)mant3 * 0.125f, ef - 7);

    return sign * val;
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
