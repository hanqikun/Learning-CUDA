#include <vector>
#include <musa_fp16.h>
#include <musa_runtime.h>   // MUSA 运行时 API
#include <cmath>
#include <algorithm>

#include "../tester/utils.h"

// ===================== 设备端类型转换 =====================
template <typename T>
__device__ inline float dev_to_float(T val) {
    return static_cast<float>(val);
}
template <>
__device__ inline float dev_to_float<half>(half val) {
    return __half2float(val);
}

template <typename T>
__device__ inline T dev_from_float(float val) {
    return static_cast<T>(val);
}
template <>
__device__ inline half dev_from_float<half>(float val) {
    return __float2half(val);
}

// ===================== RMSNorm GPU Kernel =====================
template <typename T>
__global__ void rmsNormKernel(
    const T* __restrict__ input,   // [rows, hidden_dim]
    const T* __restrict__ weight,  // [hidden_dim]
    T* __restrict__ output,        // [rows, hidden_dim]
    int rows,
    int hidden_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    float* partial_sums = smem;    // 长度 blockDim.x

    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    // ---------- 第一步：计算该行所有元素的平方和 ----------
    float local_sum = 0.0f;
    for (int j = tid; j < hidden_dim; j += total_threads) {
        float val = dev_to_float<T>(input[row * hidden_dim + j]);
        local_sum += val * val;
    }
    partial_sums[tid] = local_sum;
    __syncthreads();

    // 共享内存归约
    for (int offset = total_threads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial_sums[tid] += partial_sums[tid + offset];
        }
        __syncthreads();
    }

    float sum_sq = partial_sums[0];
    float rms = sqrtf(sum_sq / static_cast<float>(hidden_dim) + eps);

    // ---------- 第二步：归一化并乘以权重 ----------
    for (int j = tid; j < hidden_dim; j += total_threads) {
        float val = dev_to_float<T>(input[row * hidden_dim + j]);
        float w = dev_to_float<T>(weight[j]);
        float out_val = val * w / rms;
        output[row * hidden_dim + j] = dev_from_float<T>(out_val);
    }
}

// ===================== 主机端 rmsNorm（GPU 实现） =====================
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
    size_t total_elements = rows * hidden_dim;
    if (h_output.size() != total_elements) {
        h_output.resize(total_elements);
    }

    T *d_input, *d_weight, *d_output;
    musaMalloc(&d_input, total_elements * sizeof(T));
    musaMalloc(&d_weight, hidden_dim * sizeof(T));
    musaMalloc(&d_output, total_elements * sizeof(T));

    musaMemcpy(d_input, h_input.data(), total_elements * sizeof(T), musaMemcpyHostToDevice);
    musaMemcpy(d_weight, h_weight.data(), hidden_dim * sizeof(T), musaMemcpyHostToDevice);

    int block_size = 256;
    while (block_size > (int)hidden_dim) block_size >>= 1;
    block_size = std::max(block_size, 1);
    dim3 block(block_size);
    dim3 grid((int)rows);

    size_t smem_bytes = block_size * sizeof(float);
    rmsNormKernel<T><<<grid, block, smem_bytes>>>(
        d_input, d_weight, d_output, (int)rows, (int)hidden_dim, eps
    );
    musaDeviceSynchronize();

    musaMemcpy(h_output.data(), d_output, total_elements * sizeof(T), musaMemcpyDeviceToHost);

    musaFree(d_input);
    musaFree(d_weight);
    musaFree(d_output);
}

// ===================== Flash Attention Kernel =====================
template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ q,   // [batch, target_seq, query_heads, head_dim]
    const T* __restrict__ k,   // [batch, src_seq,   kv_heads,    head_dim]
    const T* __restrict__ v,   // [batch, src_seq,   kv_heads,    head_dim]
    T* __restrict__ o,         // [batch, target_seq, query_heads, head_dim]
    int batch_size,
    int target_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal,
    int groups,                // query_heads / kv_heads
    int Br,
    int Bc
) {
    extern __shared__ float smem[];
    float* smem_k = smem;
    float* smem_v = smem + Bc * head_dim;

    int total_q_blocks = (target_seq_len + Br - 1) / Br;
    int blocks_per_head = total_q_blocks;
    int blocks_per_batch = query_heads * blocks_per_head;

    int block_id = blockIdx.x;
    int b = block_id / blocks_per_batch;
    int rem = block_id % blocks_per_batch;
    int qh = rem / blocks_per_head;
    int q_start = (rem % blocks_per_head) * Br;

    int tid = threadIdx.x;
    int i = tid;
    bool valid_row = (i < Br && (q_start + i) < target_seq_len);
    int global_i = q_start + i;
    int kv_h = qh / groups;

    int q_base = ((b * target_seq_len + global_i) * query_heads + qh) * head_dim;

    float q_reg[256];  // 假设 head_dim <= 256
    if (valid_row) {
        for (int d = 0; d < head_dim; ++d) {
            q_reg[d] = dev_to_float<T>(q[q_base + d]);
        }
    }

    float m = -INFINITY;
    float l = 0.0f;
    float O[256] = {0.0f};
    float scale = rsqrtf((float)head_dim);

    // ---------- 外循环：遍历所有 K/V 块 ----------
    for (int kv_start = 0; kv_start < src_seq_len; kv_start += Bc) {
        int kv_len = min(Bc, src_seq_len - kv_start);

        // 1. 协作加载 K/V 到共享内存
        for (int idx = tid; idx < kv_len * head_dim; idx += blockDim.x) {
            int j = idx / head_dim;
            int d = idx % head_dim;
            int global_j = kv_start + j;
            int k_idx = ((b * src_seq_len + global_j) * kv_heads + kv_h) * head_dim + d;
            int v_idx = ((b * src_seq_len + global_j) * kv_heads + kv_h) * head_dim + d;
            smem_k[j * head_dim + d] = dev_to_float<T>(k[k_idx]);
            smem_v[j * head_dim + d] = dev_to_float<T>(v[v_idx]);
        }
        __syncthreads();

        if (valid_row) {
            // 第一遍：当前块内最大值
            float row_max = -INFINITY;
            for (int j = 0; j < kv_len; ++j) {
                int global_j = kv_start + j;
                if (is_causal && global_j > global_i) continue;
                float s = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    s += q_reg[d] * smem_k[j * head_dim + d];
                }
                s *= scale;
                if (s > row_max) row_max = s;
            }

            if (row_max != -INFINITY) {
                float m_prev = m;
                float m_new = fmaxf(m_prev, row_max);

                // 第二遍：计算局部指数和与加权 V
                float sum_exp = 0.0f;
                float P_block[256] = {0.0f};
                for (int j = 0; j < kv_len; ++j) {
                    int global_j = kv_start + j;
                    if (is_causal && global_j > global_i) continue;
                    float s = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        s += q_reg[d] * smem_k[j * head_dim + d];
                    }
                    s *= scale;
                    float p = expf(s - m_new);
                    sum_exp += p;
                    for (int d = 0; d < head_dim; ++d) {
                        P_block[d] += p * smem_v[j * head_dim + d];
                    }
                }

                // 在线 softmax 更新
                float l_old_scaled = l * expf(m_prev - m_new);
                float l_new = l_old_scaled + sum_exp;
                float inv_l_new = (l_new > 0.0f) ? 1.0f / l_new : 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    O[d] = (l_old_scaled * O[d] + P_block[d]) * inv_l_new;
                }

                m = m_new;
                l = l_new;
            }
        }
        __syncthreads();
    }

    // 写回输出
    if (valid_row) {
        int o_base = ((b * target_seq_len + global_i) * query_heads + qh) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            o[o_base + d] = dev_from_float<T>(O[d]);
        }
    }
}

// ===================== 主机端 flashAttention 接口 =====================
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    size_t q_size = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    size_t kv_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = q_size;

    if (h_o.size() != o_size) {
        h_o.resize(o_size);
    }

    T *d_q, *d_k, *d_v, *d_o;
    musaMalloc(&d_q, q_size * sizeof(T));
    musaMalloc(&d_k, kv_size * sizeof(T));
    musaMalloc(&d_v, kv_size * sizeof(T));
    musaMalloc(&d_o, o_size * sizeof(T));

    musaMemcpy(d_q, h_q.data(), q_size * sizeof(T), musaMemcpyHostToDevice);
    musaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), musaMemcpyHostToDevice);
    musaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), musaMemcpyHostToDevice);

    // 分块参数：Br 固定 128，Bc 根据共享内存大小动态计算
    const int Br = 128;
    int Bc;
    // 查询设备最大共享内存
    int max_shmem;
    musaDeviceGetAttribute(&max_shmem, musaDevAttrMaxSharedMemoryPerBlock, 0);  // 修正枚举名
    Bc = (int)(max_shmem / (2 * head_dim * sizeof(float)));
    Bc = std::max(1, std::min(Bc, 64));
    Bc = std::min(Bc, src_seq_len);
    Bc = (Bc / 4) * 4;          // 对齐
    if (Bc == 0) Bc = 4;

    int total_q_blocks = (target_seq_len + Br - 1) / Br;
    int blocks_per_batch = query_heads * total_q_blocks;
    int grid_size = batch_size * blocks_per_batch;
    dim3 block(Br);
    dim3 grid(grid_size);

    size_t shmem_bytes = (size_t)2 * Bc * head_dim * sizeof(float);
    int groups = query_heads / kv_heads;

    flashAttentionKernel<T><<<grid, block, shmem_bytes>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal,
        groups,
        Br, Bc
    );
    musaDeviceSynchronize();

    musaMemcpy(h_o.data(), d_o, o_size * sizeof(T), musaMemcpyDeviceToHost);

    musaFree(d_q);
    musaFree(d_k);
    musaFree(d_v);
    musaFree(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
