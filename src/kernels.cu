#pragma once

#include <iostream>
#include <cstdlib>          // for exit
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

// ===================== 平台适配与 RUNTIME_CHECK =====================
#if defined(PLATFORM_ILUVATAR)
    // 天数平台兼容 CUDA API
    #define RUNTIME_ERR_TYPE cudaError_t
    #define RUNTIME_SUCCESS_CODE cudaSuccess
    #define RUNTIME_GET_ERROR_STR cudaGetErrorString

    #define MY_MALLOC cudaMalloc
    #define MY_FREE cudaFree
    #define MY_MEMCPY cudaMemcpy
    #define MY_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
    #define MY_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
    #define MY_DEVICE_SYNCHRONIZE cudaDeviceSynchronize

    // 保留平台特有的默认分块参数（当无法获取属性时作为备选，但我们会动态获取）
    #define FLASH_Br 64   // 对于天数卡，Br 仍设为 64（可根据需要调整）

#elif defined(PLATFORM_NVIDIA)
    #define RUNTIME_ERR_TYPE cudaError_t
    #define RUNTIME_SUCCESS_CODE cudaSuccess
    #define RUNTIME_GET_ERROR_STR cudaGetErrorString

    #define MY_MALLOC cudaMalloc
    #define MY_FREE cudaFree
    #define MY_MEMCPY cudaMemcpy
    #define MY_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
    #define MY_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
    #define MY_DEVICE_SYNCHRONIZE cudaDeviceSynchronize

    #define FLASH_Br 128

#else
    #error "Unsupported PLATFORM. Define PLATFORM_NVIDIA or PLATFORM_ILUVATAR."
#endif

#define RUNTIME_CHECK(call)                                                    \
  do {                                                                         \
    RUNTIME_ERR_TYPE err = call;                                               \
    if (err != RUNTIME_SUCCESS_CODE) {                                         \
      std::cerr << "Runtime error at " << __FILE__ << ":" << __LINE__ << " - " \
                << RUNTIME_GET_ERROR_STR(err) << "\n";                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ===================== 设备端类型转换（不变） =====================
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

// ===================== RMSNorm GPU Kernel（不变） =====================
template <typename T>
__global__ void rmsNormKernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    T* __restrict__ output,
    int rows,
    int hidden_dim,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    float* partial_sums = smem;

    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    float local_sum = 0.0f;
    for (int j = tid; j < hidden_dim; j += total_threads) {
        float val = dev_to_float<T>(input[row * hidden_dim + j]);
        local_sum += val * val;
    }
    partial_sums[tid] = local_sum;
    __syncthreads();

    for (int offset = total_threads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial_sums[tid] += partial_sums[tid + offset];
        }
        __syncthreads();
    }

    float sum_sq = partial_sums[0];
    float rms = sqrtf(sum_sq / static_cast<float>(hidden_dim) + eps);

    for (int j = tid; j < hidden_dim; j += total_threads) {
        float val = dev_to_float<T>(input[row * hidden_dim + j]);
        float w = dev_to_float<T>(weight[j]);
        float out_val = val * w / rms;
        output[row * hidden_dim + j] = dev_from_float<T>(out_val);
    }
}

// ===================== 主机端 rmsNorm（不变） =====================
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
             std::vector<T>& h_output, size_t rows, size_t hidden_dim,
             float eps) {
    size_t total_elements = rows * hidden_dim;

    T *d_input, *d_weight, *d_output;
    RUNTIME_CHECK(MY_MALLOC(&d_input, total_elements * sizeof(T)));
    RUNTIME_CHECK(MY_MALLOC(&d_weight, hidden_dim * sizeof(T)));
    RUNTIME_CHECK(MY_MALLOC(&d_output, total_elements * sizeof(T)));

    RUNTIME_CHECK(MY_MEMCPY(d_input, h_input.data(), total_elements * sizeof(T), MY_MEMCPY_HOST_TO_DEVICE));
    RUNTIME_CHECK(MY_MEMCPY(d_weight, h_weight.data(), hidden_dim * sizeof(T), MY_MEMCPY_HOST_TO_DEVICE));

    int block_size = 256;
    while (block_size > hidden_dim) block_size /= 2;
    block_size = std::max(block_size, 1);
    dim3 block(block_size);
    dim3 grid(rows);

    size_t smem_bytes = block_size * sizeof(float);
    rmsNormKernel<T><<<grid, block, smem_bytes>>>(
        d_input, d_weight, d_output, rows, hidden_dim, eps
    );
    RUNTIME_CHECK(MY_DEVICE_SYNCHRONIZE());

    RUNTIME_CHECK(MY_MEMCPY(h_output.data(), d_output, total_elements * sizeof(T), MY_MEMCPY_DEVICE_TO_HOST));

    RUNTIME_CHECK(MY_FREE(d_input));
    RUNTIME_CHECK(MY_FREE(d_weight));
    RUNTIME_CHECK(MY_FREE(d_output));
}

// ===================== Flash Attention Kernel（不变） =====================
template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ o,
    int batch_size,
    int target_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal,
    int groups,
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
    bool valid_row = (i < min(Br, target_seq_len - q_start));
    int global_i = q_start + i;
    int kv_h = qh / groups;

    int q_base = ((b * target_seq_len + global_i) * query_heads + qh) * head_dim;

    const int MAX_DIM = 256;
    float q_reg[MAX_DIM];
    if (valid_row) {
        for (int d = 0; d < head_dim; ++d) {
            q_reg[d] = dev_to_float<T>(q[q_base + d]);
        }
    }

    float m = -INFINITY;
    float l = 0.0f;
    float O[MAX_DIM];
    for (int d = 0; d < head_dim; ++d) O[d] = 0.0f;

    float scale = rsqrtf((float)head_dim);

    for (int kv_start = 0; kv_start < src_seq_len; kv_start += Bc) {
        int kv_len = min(Bc, src_seq_len - kv_start);

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

                float sum_exp = 0.0f;
                float P_block[MAX_DIM] = {0.0f};
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

    if (valid_row) {
        int o_base = ((b * target_seq_len + global_i) * query_heads + qh) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            o[o_base + d] = dev_from_float<T>(O[d]);
        }
    }
}

// ===================== 主机端 flashAttention（动态共享内存适配） =====================
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    size_t q_size = (size_t)batch_size * target_seq_len * query_heads * head_dim;
    size_t kv_size = (size_t)batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = q_size;

    T *d_q, *d_k, *d_v, *d_o;
    RUNTIME_CHECK(MY_MALLOC(&d_q, q_size * sizeof(T)));
    RUNTIME_CHECK(MY_MALLOC(&d_k, kv_size * sizeof(T)));
    RUNTIME_CHECK(MY_MALLOC(&d_v, kv_size * sizeof(T)));
    RUNTIME_CHECK(MY_MALLOC(&d_o, o_size * sizeof(T)));

    RUNTIME_CHECK(MY_MEMCPY(d_q, h_q.data(), q_size * sizeof(T), MY_MEMCPY_HOST_TO_DEVICE));
    RUNTIME_CHECK(MY_MEMCPY(d_k, h_k.data(), kv_size * sizeof(T), MY_MEMCPY_HOST_TO_DEVICE));
    RUNTIME_CHECK(MY_MEMCPY(d_v, h_v.data(), kv_size * sizeof(T), MY_MEMCPY_HOST_TO_DEVICE));

    // ---------- 动态获取设备最大共享内存 ----------
    int device;
    RUNTIME_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    RUNTIME_CHECK(cudaGetDeviceProperties(&props, device));
    size_t max_shmem = props.sharedMemPerBlock;   // 实际硬件支持的最大共享内存 / block

    // ---------- 根据共享内存大小计算最优 Bc ----------
    const int Br = FLASH_Br;  // 保持平台特定的行块大小
    int Bc;
    // 每个线程块需要的共享内存：K 和 V 各占 Bc * head_dim * sizeof(float)
    Bc = (int)(max_shmem / (2 * head_dim * sizeof(float)));
    // 限制 Bc 范围：至少为 1，最多 64（可调，但需考虑寄存器压力），且不能超过 src_seq_len
    Bc = std::max(1, std::min(Bc, 64));
    Bc = std::min(Bc, src_seq_len);
    // 对齐到 4 的倍数（有利于内存合并）
    Bc = (Bc / 4) * 4;
    if (Bc == 0) Bc = 4;

    // 如果实际需要的共享内存超过最大限制，则降低 Bc（但上述计算已保证不超）
    size_t required_shmem = (size_t)2 * Bc * head_dim * sizeof(float);
    if (required_shmem > max_shmem) {
        // 保险处理：逐步减小 Bc 直到满足
        while (required_shmem > max_shmem && Bc > 1) {
            Bc -= 4;
            required_shmem = (size_t)2 * Bc * head_dim * sizeof(float);
        }
    }

    // 启动 Kernel
    int total_q_blocks = (target_seq_len + Br - 1) / Br;
    int blocks_per_batch = query_heads * total_q_blocks;
    int grid_size = batch_size * blocks_per_batch;
    dim3 block(Br);
    dim3 grid(grid_size);

    size_t shmem_bytes = required_shmem;  // 实际使用的共享内存大小
    flashAttentionKernel<T><<<grid, block, shmem_bytes>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        is_causal,
        query_heads / kv_heads,
        Br, Bc
    );
    RUNTIME_CHECK(MY_DEVICE_SYNCHRONIZE());

    RUNTIME_CHECK(MY_MEMCPY(h_o.data(), d_o, o_size * sizeof(T), MY_MEMCPY_DEVICE_TO_HOST));

    RUNTIME_CHECK(MY_FREE(d_q));
    RUNTIME_CHECK(MY_FREE(d_k));
    RUNTIME_CHECK(MY_FREE(d_v));
    RUNTIME_CHECK(MY_FREE(d_o));
}

// ===================== 显式模板实例化（不变） =====================
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
