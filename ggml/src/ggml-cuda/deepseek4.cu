#include "deepseek4.cuh"
#include "ggml.h"

// ============================================================================
// OP 1: dsv4_hc_split_sinkhorn
// ============================================================================

__global__ void dsv4_hc_split_sinkhorn_f32(
        const float * __restrict__ mixes,
        const float * __restrict__ scale_data,
        const float * __restrict__ base_data,
        float * __restrict__ dst,
        const int n_hc,
        const int sinkhorn_iters,
        const float eps,
        const int64_t n_rows,
        const int64_t mix_hc,
        const int64_t nb_mix1,
        const int64_t nb_dst1) {

    const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }

    const float * mix = mixes + row * nb_mix1;
    float * out = dst + row * nb_dst1;

    const float pre_scale  = scale_data[0];
    const float post_scale = scale_data[1];
    const float comb_scale = scale_data[2];

    if (n_hc == 4) {
        const float4 pre_z = make_float4(
            mix[0] * pre_scale + base_data[0],
            mix[1] * pre_scale + base_data[1],
            mix[2] * pre_scale + base_data[2],
            mix[3] * pre_scale + base_data[3]);
        out[0] = 1.0f / (1.0f + expf(-pre_z.x)) + eps;
        out[1] = 1.0f / (1.0f + expf(-pre_z.y)) + eps;
        out[2] = 1.0f / (1.0f + expf(-pre_z.z)) + eps;
        out[3] = 1.0f / (1.0f + expf(-pre_z.w)) + eps;

        const float4 post_z = make_float4(
            mix[4] * post_scale + base_data[4],
            mix[5] * post_scale + base_data[5],
            mix[6] * post_scale + base_data[6],
            mix[7] * post_scale + base_data[7]);
        out[4] = 2.0f / (1.0f + expf(-post_z.x));
        out[5] = 2.0f / (1.0f + expf(-post_z.y));
        out[6] = 2.0f / (1.0f + expf(-post_z.z));
        out[7] = 2.0f / (1.0f + expf(-post_z.w));

        float4 r0 = make_float4(
            mix[ 8] * comb_scale + base_data[ 8],
            mix[ 9] * comb_scale + base_data[ 9],
            mix[10] * comb_scale + base_data[10],
            mix[11] * comb_scale + base_data[11]);
        float4 r1 = make_float4(
            mix[12] * comb_scale + base_data[12],
            mix[13] * comb_scale + base_data[13],
            mix[14] * comb_scale + base_data[14],
            mix[15] * comb_scale + base_data[15]);
        float4 r2 = make_float4(
            mix[16] * comb_scale + base_data[16],
            mix[17] * comb_scale + base_data[17],
            mix[18] * comb_scale + base_data[18],
            mix[19] * comb_scale + base_data[19]);
        float4 r3 = make_float4(
            mix[20] * comb_scale + base_data[20],
            mix[21] * comb_scale + base_data[21],
            mix[22] * comb_scale + base_data[22],
            mix[23] * comb_scale + base_data[23]);

        const float m0 = fmaxf(fmaxf(r0.x, r0.y), fmaxf(r0.z, r0.w));
        const float m1 = fmaxf(fmaxf(r1.x, r1.y), fmaxf(r1.z, r1.w));
        const float m2 = fmaxf(fmaxf(r2.x, r2.y), fmaxf(r2.z, r2.w));
        const float m3 = fmaxf(fmaxf(r3.x, r3.y), fmaxf(r3.z, r3.w));

        r0 = make_float4(expf(r0.x - m0), expf(r0.y - m0), expf(r0.z - m0), expf(r0.w - m0));
        r1 = make_float4(expf(r1.x - m1), expf(r1.y - m1), expf(r1.z - m1), expf(r1.w - m1));
        r2 = make_float4(expf(r2.x - m2), expf(r2.y - m2), expf(r2.z - m2), expf(r2.w - m2));
        r3 = make_float4(expf(r3.x - m3), expf(r3.y - m3), expf(r3.z - m3), expf(r3.w - m3));

        const float inv_sum0 = 1.0f / (r0.x + r0.y + r0.z + r0.w);
        const float inv_sum1 = 1.0f / (r1.x + r1.y + r1.z + r1.w);
        const float inv_sum2 = 1.0f / (r2.x + r2.y + r2.z + r2.w);
        const float inv_sum3 = 1.0f / (r3.x + r3.y + r3.z + r3.w);

        r0 = make_float4(r0.x * inv_sum0 + eps, r0.y * inv_sum0 + eps, r0.z * inv_sum0 + eps, r0.w * inv_sum0 + eps);
        r1 = make_float4(r1.x * inv_sum1 + eps, r1.y * inv_sum1 + eps, r1.z * inv_sum1 + eps, r1.w * inv_sum1 + eps);
        r2 = make_float4(r2.x * inv_sum2 + eps, r2.y * inv_sum2 + eps, r2.z * inv_sum2 + eps, r2.w * inv_sum2 + eps);
        r3 = make_float4(r3.x * inv_sum3 + eps, r3.y * inv_sum3 + eps, r3.z * inv_sum3 + eps, r3.w * inv_sum3 + eps);

        float4 col_inv = make_float4(
            1.0f / (r0.x + r1.x + r2.x + r3.x + eps),
            1.0f / (r0.y + r1.y + r2.y + r3.y + eps),
            1.0f / (r0.z + r1.z + r2.z + r3.z + eps),
            1.0f / (r0.w + r1.w + r2.w + r3.w + eps));
        r0 = make_float4(r0.x * col_inv.x, r0.y * col_inv.y, r0.z * col_inv.z, r0.w * col_inv.w);
        r1 = make_float4(r1.x * col_inv.x, r1.y * col_inv.y, r1.z * col_inv.z, r1.w * col_inv.w);
        r2 = make_float4(r2.x * col_inv.x, r2.y * col_inv.y, r2.z * col_inv.z, r2.w * col_inv.w);
        r3 = make_float4(r3.x * col_inv.x, r3.y * col_inv.y, r3.z * col_inv.z, r3.w * col_inv.w);

        for (int iter = 1; iter < sinkhorn_iters; ++iter) {
            r0 = make_float4(
                r0.x / (r0.x + r0.y + r0.z + r0.w + eps),
                r0.y / (r0.x + r0.y + r0.z + r0.w + eps),
                r0.z / (r0.x + r0.y + r0.z + r0.w + eps),
                r0.w / (r0.x + r0.y + r0.z + r0.w + eps));
            r1 = make_float4(
                r1.x / (r1.x + r1.y + r1.z + r1.w + eps),
                r1.y / (r1.x + r1.y + r1.z + r1.w + eps),
                r1.z / (r1.x + r1.y + r1.z + r1.w + eps),
                r1.w / (r1.x + r1.y + r1.z + r1.w + eps));
            r2 = make_float4(
                r2.x / (r2.x + r2.y + r2.z + r2.w + eps),
                r2.y / (r2.x + r2.y + r2.z + r2.w + eps),
                r2.z / (r2.x + r2.y + r2.z + r2.w + eps),
                r2.w / (r2.x + r2.y + r2.z + r2.w + eps));
            r3 = make_float4(
                r3.x / (r3.x + r3.y + r3.z + r3.w + eps),
                r3.y / (r3.x + r3.y + r3.z + r3.w + eps),
                r3.z / (r3.x + r3.y + r3.z + r3.w + eps),
                r3.w / (r3.x + r3.y + r3.z + r3.w + eps));

            col_inv = make_float4(
                1.0f / (r0.x + r1.x + r2.x + r3.x + eps),
                1.0f / (r0.y + r1.y + r2.y + r3.y + eps),
                1.0f / (r0.z + r1.z + r2.z + r3.z + eps),
                1.0f / (r0.w + r1.w + r2.w + r3.w + eps));
            r0 = make_float4(r0.x * col_inv.x, r0.y * col_inv.y, r0.z * col_inv.z, r0.w * col_inv.w);
            r1 = make_float4(r1.x * col_inv.x, r1.y * col_inv.y, r1.z * col_inv.z, r1.w * col_inv.w);
            r2 = make_float4(r2.x * col_inv.x, r2.y * col_inv.y, r2.z * col_inv.z, r2.w * col_inv.w);
            r3 = make_float4(r3.x * col_inv.x, r3.y * col_inv.y, r3.z * col_inv.z, r3.w * col_inv.w);
        }

        out[ 8] = r0.x; out[ 9] = r0.y; out[10] = r0.z; out[11] = r0.w;
        out[12] = r1.x; out[13] = r1.y; out[14] = r1.z; out[15] = r1.w;
        out[16] = r2.x; out[17] = r2.y; out[18] = r2.z; out[19] = r2.w;
        out[20] = r3.x; out[21] = r3.y; out[22] = r3.z; out[23] = r3.w;
        return;
    }

    for (int i = 0; i < n_hc; ++i) {
        const float z = mix[i] * pre_scale + base_data[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }

    for (int i = 0; i < n_hc; ++i) {
        const int off = n_hc + i;
        const float z = mix[off] * post_scale + base_data[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[16*16];

    for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            const int off = 2 * n_hc + idx;
            const float v = mix[off] * comb_scale + base_data[off];
            c[idx] = v;
            row_max = fmaxf(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            const int idx = src_hc + dst_hc * n_hc;
            c[idx] = c[idx] * inv_sum + eps;
        }
    }

    for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            sum += c[src_hc + dst_hc * n_hc];
        }
        const float inv_denom = 1.0f / (sum + eps);
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            c[src_hc + dst_hc * n_hc] *= inv_denom;
        }
    }

    for (int iter = 1; iter < sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < n_hc * n_hc; ++i) {
        out[2 * n_hc + i] = c[i];
    }
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(mixes->nb[0] == sizeof(float));

    const int n_hc           = ggml_get_op_params_i32(dst, 0);
    const int sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float eps          = ggml_get_op_params_f32(dst, 2);
    const int64_t mix_hc     = mixes->ne[0];
    const int64_t n_rows     = ggml_nrows(mixes);

    GGML_ASSERT(n_hc > 0 && n_hc <= 16);
    GGML_ASSERT(sinkhorn_iters > 0);
    GGML_ASSERT(mix_hc == (2 + n_hc) * n_hc);

    cudaStream_t stream = ctx.stream();

    const int block_size = 256;
    const int grid_size  = (n_rows + block_size - 1) / block_size;

    dsv4_hc_split_sinkhorn_f32<<<grid_size, block_size, 0, stream>>>(
        (const float *) mixes->data,
        (const float *) scale->data,
        (const float *) base->data,
        (float *) dst->data,
        n_hc, sinkhorn_iters, eps, n_rows, mix_hc,
        mixes->nb[1] / sizeof(float),
        dst->nb[1] / sizeof(float));
}

// ============================================================================
// OP 2: dsv4_hc_weighted_sum
// ============================================================================

__global__ void dsv4_hc_weighted_sum_f32(
        const char * __restrict__ x,
        const char * __restrict__ weights,
        char * __restrict__ dst,
        const int64_t n_embd,
        const int64_t n_hc,
        const int64_t n_tokens,
        const int64_t nb_x0, const int64_t nb_x1, const int64_t nb_x2,
        const int64_t nb_w0, const int64_t nb_w1,
        const int64_t nb_d0, const int64_t nb_d1) {

    const int64_t n_elem = n_embd * n_tokens;
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elem) {
        return;
    }

    const int64_t d = i % n_embd;
    const int64_t t = i / n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < n_hc; ++h) {
        const float xv = *(const float *) (x + d * nb_x0 + h * nb_x1 + t * nb_x2);
        const float wv = *(const float *) (weights + h * nb_w0 + t * nb_w1);
        acc += xv * wv;
    }

    *(float *) (dst + d * nb_d0 + t * nb_d1) = acc;
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type       == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type     == GGML_TYPE_F32);
    GGML_ASSERT(x->ne[0]       == dst->ne[0]);
    GGML_ASSERT(x->ne[1]       == weights->ne[0]);
    GGML_ASSERT(x->ne[2]       == dst->ne[1]);
    GGML_ASSERT(weights->ne[1] == dst->ne[1]);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_hc     = x->ne[1];
    const int64_t n_tokens = dst->ne[1];
    const int64_t n_elem   = n_embd * n_tokens;

    cudaStream_t stream = ctx.stream();

    const int block_size = 256;
    const int grid_size  = (n_elem + block_size - 1) / block_size;

    dsv4_hc_weighted_sum_f32<<<grid_size, block_size, 0, stream>>>(
        (const char *) x->data,
        (const char *) weights->data,
        (char *) dst->data,
        n_embd, n_hc, n_tokens,
        x->nb[0], x->nb[1], x->nb[2],
        weights->nb[0], weights->nb[1],
        dst->nb[0], dst->nb[1]);
}

// ============================================================================
// OP 3: dsv4_hc_expand
// ============================================================================

__global__ void dsv4_hc_expand_f32(
        const char * __restrict__ block_out,
        const char * __restrict__ residual,
        const char * __restrict__ post,
        const char * __restrict__ comb,
        char * __restrict__ dst,
        const int64_t n_embd,
        const int64_t n_hc,
        const int64_t n_tokens,
        const int64_t nb_block0, const int64_t nb_block1,
        const int64_t nb_res0, const int64_t nb_res1, const int64_t nb_res2,
        const int64_t nb_post0, const int64_t nb_post1,
        const int64_t nb_comb0, const int64_t nb_comb1, const int64_t nb_comb2,
        const int64_t nb_d0, const int64_t nb_d1, const int64_t nb_d2) {

    const int64_t n_elem = n_embd * n_hc * n_tokens;
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elem) {
        return;
    }

    const int64_t d      = i % n_embd;
    const int64_t tmp    = i / n_embd;
    const int64_t dst_hc = tmp % n_hc;
    const int64_t t      = tmp / n_hc;

    const float block_v = *(const float *) (block_out + d * nb_block0 + t * nb_block1);
    const float post_v  = *(const float *) (post      + dst_hc * nb_post0 + t * nb_post1);

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < n_hc; ++src_hc) {
        const float comb_v = *(const float *) (comb     + dst_hc * nb_comb0 + src_hc * nb_comb1 + t * nb_comb2);
        const float res_v  = *(const float *) (residual + d * nb_res0 + src_hc * nb_res1 + t * nb_res2);
        acc += comb_v * res_v;
    }

    *(float *) (dst + d * nb_d0 + dst_hc * nb_d1 + t * nb_d2) = acc;
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * block_out = dst->src[0];
    const ggml_tensor * residual  = dst->src[1];
    const ggml_tensor * post      = dst->src[2];
    const ggml_tensor * comb      = dst->src[3];

    GGML_ASSERT(block_out->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type  == GGML_TYPE_F32);
    GGML_ASSERT(post->type      == GGML_TYPE_F32);
    GGML_ASSERT(comb->type      == GGML_TYPE_F32);
    GGML_ASSERT(dst->type       == GGML_TYPE_F32);
    GGML_ASSERT(block_out->ne[0] == dst->ne[0]);
    GGML_ASSERT(block_out->ne[1] == dst->ne[2]);
    GGML_ASSERT(residual->ne[0]  == dst->ne[0]);
    GGML_ASSERT(residual->ne[1]  == dst->ne[1]);
    GGML_ASSERT(residual->ne[2]  == dst->ne[2]);
    GGML_ASSERT(post->ne[0]      == dst->ne[1]);
    GGML_ASSERT(post->ne[1]      == dst->ne[2]);
    GGML_ASSERT(comb->ne[0]      == dst->ne[1]);
    GGML_ASSERT(comb->ne[1]      == dst->ne[1]);
    GGML_ASSERT(comb->ne[2]      == dst->ne[2]);

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_hc     = dst->ne[1];
    const int64_t n_tokens = dst->ne[2];
    const int64_t n_elem   = n_embd * n_hc * n_tokens;

    cudaStream_t stream = ctx.stream();

    const int block_size = 256;
    const int grid_size  = (n_elem + block_size - 1) / block_size;

    dsv4_hc_expand_f32<<<grid_size, block_size, 0, stream>>>(
        (const char *) block_out->data,
        (const char *) residual->data,
        (const char *) post->data,
        (const char *) comb->data,
        (char *) dst->data,
        n_embd, n_hc, n_tokens,
        block_out->nb[0], block_out->nb[1],
        residual->nb[0], residual->nb[1], residual->nb[2],
        post->nb[0], post->nb[1],
        comb->nb[0], comb->nb[1], comb->nb[2],
        dst->nb[0], dst->nb[1], dst->nb[2]);
}

// ============================================================================
// OP 4: dsv4_fp8_kv_quantize
// ============================================================================

__global__ void dsv4_fp8_kv_quantize_f32(
        const char * __restrict__ src0,
        char * __restrict__ dst,
        const int64_t head_dim,
        const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb_d0, const int64_t nb_d1, const int64_t nb_d2, const int64_t nb_d3,
        const int64_t n_rot,
        const int64_t n_nope) {

    const int64_t n_rows = ne01 * ne02 * ne03;
    const int64_t row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }

    const int64_t i1 = row % ne01;
    const int64_t i2 = (row / ne01) % ne02;
    const int64_t i3 = row / (ne01 * ne02);

    const char * src_row = src0 + i1 * nb01 + i2 * nb02 + i3 * nb03;
    char * dst_row = dst + i1 * nb_d1 + i2 * nb_d2 + i3 * nb_d3;

    extern __shared__ float scratch[];

    for (int64_t off = 0; off < n_nope; off += 64) {
        const int tid = threadIdx.x;
        float v = 0.0f;
        if (tid < 64) {
            v = *(const float *) (src_row + (off + tid) * nb00);
            scratch[tid] = fabsf(v);
        }
        __syncthreads();

        for (uint stride = 32; stride > 0; stride >>= 1) {
            if (tid < stride) {
                scratch[tid] = fmaxf(scratch[tid], scratch[tid + stride]);
            }
            __syncthreads();
        }

        const float amax = fmaxf(scratch[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        if (tid < 64) {
            const float clamped = fminf(fmaxf(v / scale, -448.0f), 448.0f);
            const float q = dsv4_e4m3fn_dequant(clamped) * scale;
            *(float *) (dst_row + (off + tid) * nb_d0) = q;
        }
        __syncthreads();
    }

    for (int64_t i = n_nope + threadIdx.x; i < head_dim; i += 64) {
        *(float *) (dst_row + i * nb_d0) = *(const float *) (src_row + i * nb00);
    }
}

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t n_rot    = ggml_get_op_params_i32(dst, 0);
    const int64_t head_dim = src0->ne[0];
    const int64_t n_nope   = head_dim - n_rot;

    GGML_ASSERT(n_rot >= 0);
    GGML_ASSERT(n_nope > 0);
    GGML_ASSERT(n_nope % 64 == 0);

    const int64_t n_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];

    cudaStream_t stream = ctx.stream();

    const int block_size = 64;
    const int grid_size  = n_rows;
    const size_t shared_mem = 64 * sizeof(float);

    dsv4_fp8_kv_quantize_f32<<<grid_size, block_size, shared_mem, stream>>>(
        (const char *) src0->data,
        (char *) dst->data,
        head_dim,
        src0->ne[1], src0->ne[2], src0->ne[3],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
        n_rot, n_nope);
}

// ============================================================================
// OP 5: dsv4_rope_tail
// ============================================================================

template <bool forward, bool has_ff>
__global__ void dsv4_rope_tail_f32(
        const char * __restrict__ src0,
        const int32_t * __restrict__ pos,
        const float * __restrict__ freq_factors,
        char * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb_d0, const int64_t nb_d1, const int64_t nb_d2, const int64_t nb_d3,
        const int n_dims,
        const int mode,
        const float theta_scale,
        const rope_corr_dims corr_dims,
        const float freq_scale,
        const float ext_factor,
        const float attn_factor) {

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    if (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        return;
    }

    const int n_nope = ne00 - n_dims;

    const char * src_row = src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    char * dst_row = dst + i3 * nb_d3 + i2 * nb_d2 + i1 * nb_d1;

    const int32_t p = pos[i2];

    // Copy nope prefix
    for (int64_t i0 = threadIdx.x; i0 < n_nope; i0 += blockDim.x) {
        *(float *) (dst_row + i0 * nb_d0) = *(const float *) (src_row + i0 * nb00);
    }

    __syncthreads();

    if (mode == GGML_ROPE_TYPE_NORMAL) {
        // Adjacent pairs: (n_nope, n_nope+1), (n_nope+2, n_nope+3), ...
        int i0 = n_nope + 2 * threadIdx.x;
        if (i0 >= ne00 - 1) {
            return;
        }

        const int ic = i0 - n_nope; // even index within rope tail

        float cos_theta, sin_theta;
        const float theta_base = p * powf(theta_scale, ic / 2.0f);

        float ff = 1.0f;
        if constexpr (has_ff) {
            ff = freq_factors[ic / 2];
        }

        rope_yarn<forward>(theta_base / ff, freq_scale, corr_dims, ic, ext_factor, attn_factor, cos_theta, sin_theta);

        const float x0 = *(const float *) (src_row + i0 * nb00);
        const float x1 = *(const float *) (src_row + (i0 + 1) * nb00);

        *(float *) (dst_row + i0 * nb_d0)         = x0 * cos_theta - x1 * sin_theta;
        *(float *) (dst_row + (i0 + 1) * nb_d0)   = x0 * sin_theta + x1 * cos_theta;
    } else {
        // NEOX: half-split pairs (n_nope + ic, n_nope + ic + n_half)
        const int n_half = n_dims / 2;
        int ic = threadIdx.x;
        if (ic >= n_half) {
            return;
        }

        float cos_theta, sin_theta;
        const float theta_base = p * powf(theta_scale, ic);

        float ff = 1.0f;
        if constexpr (has_ff) {
            ff = freq_factors[ic];
        }

        rope_yarn<forward>(theta_base / ff, freq_scale, corr_dims, 2 * ic, ext_factor, attn_factor, cos_theta, sin_theta);

        const float x0 = *(const float *) (src_row + (n_nope + ic) * nb00);
        const float x1 = *(const float *) (src_row + (n_nope + ic + n_half) * nb00);

        *(float *) (dst_row + (n_nope + ic) * nb_d0)             = x0 * cos_theta - x1 * sin_theta;
        *(float *) (dst_row + (n_nope + ic + n_half) * nb_d0)   = x0 * sin_theta + x1 * cos_theta;
    }
}

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);

    const int n_dims     = ((int32_t *) dst->op_params)[0];
    const int mode       = ((int32_t *) dst->op_params)[1];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[2];
    const int inverse    = ((int32_t *) dst->op_params)[3];

    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params + 4, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 9, sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    GGML_ASSERT(ne00 >= n_dims);
    GGML_ASSERT(n_dims % 2 == 0);
    GGML_ASSERT(mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    float corr_dims_arr[2];
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims_arr);
    const rope_corr_dims corr_dims = { { corr_dims_arr[0], corr_dims_arr[1] } };

    const float * freq_factors_ptr = nullptr;
    bool has_ff = false;
    if (src2 != nullptr) {
        GGML_ASSERT(src2->type == GGML_TYPE_F32);
        GGML_ASSERT(src2->ne[0] >= n_dims / 2);
        freq_factors_ptr = (const float *) src2->data;
        has_ff = true;
    }

    cudaStream_t stream = ctx.stream();

    const int block_size = 256;
    const dim3 grid_dims(ne01, ne02, ne03);

    const bool forward = !inverse;

    if (src0->type == GGML_TYPE_F32) {
        if (forward) {
            if (has_ff) {
                dsv4_rope_tail_f32<true, true><<<grid_dims, block_size, 0, stream>>>(
                    (const char *) src0->data, (const int32_t *) src1->data, freq_factors_ptr,
                    (char *) dst->data,
                    ne00, ne01, ne02, ne03,
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    n_dims, mode, theta_scale, corr_dims,
                    freq_scale, ext_factor, attn_factor);
            } else {
                dsv4_rope_tail_f32<true, false><<<grid_dims, block_size, 0, stream>>>(
                    (const char *) src0->data, (const int32_t *) src1->data, nullptr,
                    (char *) dst->data,
                    ne00, ne01, ne02, ne03,
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    n_dims, mode, theta_scale, corr_dims,
                    freq_scale, ext_factor, attn_factor);
            }
        } else {
            if (has_ff) {
                dsv4_rope_tail_f32<false, true><<<grid_dims, block_size, 0, stream>>>(
                    (const char *) src0->data, (const int32_t *) src1->data, freq_factors_ptr,
                    (char *) dst->data,
                    ne00, ne01, ne02, ne03,
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    n_dims, mode, theta_scale, corr_dims,
                    freq_scale, ext_factor, attn_factor);
            } else {
                dsv4_rope_tail_f32<false, false><<<grid_dims, block_size, 0, stream>>>(
                    (const char *) src0->data, (const int32_t *) src1->data, nullptr,
                    (char *) dst->data,
                    ne00, ne01, ne02, ne03,
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    n_dims, mode, theta_scale, corr_dims,
                    freq_scale, ext_factor, attn_factor);
            }
        }
    } else {
        GGML_ASSERT(src0->type == GGML_TYPE_F16);
        // F16 path: same as F32 but with half inputs
        if (forward) {
            if (has_ff) {
                dsv4_rope_tail_f32<true, true><<<grid_dims, block_size, 0, stream>>>(
                    (const char *) src0->data, (const int32_t *) src1->data, freq_factors_ptr,
                    (char *) dst->data,
                    ne00, ne01, ne02, ne03,
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    n_dims, mode, theta_scale, corr_dims,
                    freq_scale, ext_factor, attn_factor);
            } else {
                dsv4_rope_tail_f32<true, false><<<grid_dims, block_size, 0, stream>>>(
                    (const char *) src0->data, (const int32_t *) src1->data, nullptr,
                    (char *) dst->data,
                    ne00, ne01, ne02, ne03,
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    n_dims, mode, theta_scale, corr_dims,
                    freq_scale, ext_factor, attn_factor);
            }
        } else {
            if (has_ff) {
                dsv4_rope_tail_f32<false, true><<<grid_dims, block_size, 0, stream>>>(
                    (const char *) src0->data, (const int32_t *) src1->data, freq_factors_ptr,
                    (char *) dst->data,
                    ne00, ne01, ne02, ne03,
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    n_dims, mode, theta_scale, corr_dims,
                    freq_scale, ext_factor, attn_factor);
            } else {
                dsv4_rope_tail_f32<false, false><<<grid_dims, block_size, 0, stream>>>(
                    (const char *) src0->data, (const int32_t *) src1->data, nullptr,
                    (char *) dst->data,
                    ne00, ne01, ne02, ne03,
                    src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                    dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3],
                    n_dims, mode, theta_scale, corr_dims,
                    freq_scale, ext_factor, attn_factor);
            }
        }
    }
}
