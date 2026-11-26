// Compile for NVIDIA T4 (Turing, sm_75):
//   nvcc -O3 -arch=sm_75 minkowski_t4.cu -o minkowski_t4

//#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/*#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
*/
// ============================================================================
// Log2 / Exp2 Lookup Table Configuration (K = 10 -> 1024 entries, 4 KB/table)
// ============================================================================
#define BIAS_32 ((int32_t)127 << 23)
#define K 10
#define TABLE_SIZE (1 << K)
#define TABLE_MASK (TABLE_SIZE - 1)

static int32_t h_Elog[TABLE_SIZE];
static int32_t h_Eexp[TABLE_SIZE];

void init_tables(void) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        double x = (i + 0.5) / (double)TABLE_SIZE; // cell midpoint
        h_Elog[i] = (int32_t)lround((log2(1.0 + x) - x) * (1 << 23));
        h_Eexp[i] = (int32_t)lround((exp2(x) - 1.0 - x) * (1 << 23));
    }
}

// ============================================================================
// Host Fast Math Helpers
// ============================================================================
static inline int32_t fast_log2_host(float x) {
    int32_t i;
    memcpy(&i, &x, sizeof(i));
    return (i - BIAS_32) + h_Elog[(i >> (23 - K)) & TABLE_MASK];
}

static inline float fast_exp2_host(int32_t L) {
    int32_t i = L + BIAS_32 + h_Eexp[(L >> (23 - K)) & TABLE_MASK];
    float y;
    memcpy(&y, &i, sizeof(y));
    return y;
}

// Computes x^exp_q16 for x >= 0 using Q16.16 fixed-point exponent
static inline float fast_pow_host(float x, int32_t exp_q16) {
    if (x <= 1e-12f) return 0.0f;
    int32_t L = fast_log2_host(x);
    int64_t L_scaled = ((int64_t)L * (int64_t)exp_q16) >> 16;
    if (L_scaled <= -(126LL << 23)) return 0.0f;
    if (L_scaled >=  (127LL << 23)) return INFINITY;
    return fast_exp2_host((int32_t)L_scaled);
}

// ============================================================================
// Device Fast Math Helpers (Shared Memory LUTs)
// ============================================================================
/*__device__ __forceinline__ int32_t fast_log2_dev(float x, const int32_t* __restrict__ s_Elog) {
    int32_t i = __float_as_int(x);
    return (i - BIAS_32) + s_Elog[(i >> (23 - K)) & TABLE_MASK];
}

__device__ __forceinline__ float fast_exp2_dev(int32_t L, const int32_t* __restrict__ s_Eexp) {
    int32_t i = L + BIAS_32 + s_Eexp[(L >> (23 - K)) & TABLE_MASK];
    return __int_as_float(i);
}

__device__ __forceinline__ float fast_pow_dev(
    float x,
    int32_t exp_q16,
    const int32_t* __restrict__ s_Elog,
    const int32_t* __restrict__ s_Eexp)
{
    if (x <= 1e-12f) return 0.0f;
    int32_t L = fast_log2_dev(x, s_Elog);
    int64_t L_scaled = ((int64_t)L * (int64_t)exp_q16) >> 16;
    if (L_scaled <= -(126LL << 23)) return 0.0f;
    if (L_scaled >=  (127LL << 23)) return INFINITY;
    return fast_exp2_dev((int32_t)L_scaled, s_Eexp);
}*/

// ============================================================================
// 1. Standard Minkowski Distance (CPU & T4 Tiled CUDA Kernel)
// ============================================================================
void minkowski_cpu_exact(
    const float* Q, const float* R, float* D,
    int N, int M, int F, float p)
{
    const float inv_p = 1.0f / p;
    for (int i = 0; i < N; ++i) {
        const float* q_row = Q + i * F;
        for (int j = 0; j < M; ++j) {
            const float* r_row = R + j * F;
            float acc = 0.0f;
            for (int f = 0; f < F; ++f) {
                float diff = fabsf(q_row[f] - r_row[f]);
                acc += powf(diff, p);
            }
            D[i * M + j] = powf(acc, inv_p);
        }
    }
}

constexpr int TILE_M = 16;
constexpr int TILE_N = 16;
constexpr int TILE_F = 32;

/*__global__ void minkowski_kernel_exact(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    float* __restrict__ D,
    int N, int M, int F,
    float p, float inv_p)
{
    __shared__ float s_Q[TILE_M][TILE_F];
    __shared__ float s_R[TILE_N][TILE_F];

    int tx = threadIdx.x; // 0..15
    int ty = threadIdx.y; // 0..15

    int q_row = blockIdx.y * TILE_M + ty;
    int r_col = blockIdx.x * TILE_N + tx;
    int r_load_row = blockIdx.x * TILE_N + ty;

    float acc = 0.0f;

    for (int f_base = 0; f_base < F; f_base += TILE_F) {
        // Coalesced load of 16x32 tiles from Q and R (2 elements per thread)
        #pragma unroll
        for (int step = 0; step < TILE_F; step += 16) {
            int f_idx = f_base + tx + step;
            s_Q[ty][tx + step] = (q_row < N && f_idx < F) ? Q[q_row * F + f_idx] : 0.0f;
            s_R[ty][tx + step] = (r_load_row < M && f_idx < F) ? R[r_load_row * F + f_idx] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_F; ++k) {
            float diff = fabsf(s_Q[ty][k] - s_R[tx][k]);
            acc += powf(diff, p);
        }
        __syncthreads();
    }

    if (q_row < N && r_col < M) {
        D[q_row * M + r_col] = powf(acc, inv_p);
    }
}*/

// ============================================================================
// 2. Fast Log-Domain LUT Minkowski Distance (CPU & T4 Tiled CUDA Kernel)
// ============================================================================
void minkowski_cpu_fast_lut(
    const float* Q, const float* R, float* D,
    int N, int M, int F, float p)
{
    const int32_t p_q16     = (int32_t)lround((double)p * 65536.0);
    const int32_t inv_p_q16 = (int32_t)lround((1.0 / (double)p) * 65536.0);

    for (int i = 0; i < N; ++i) {
        const float* q_row = Q + i * F;
        for (int j = 0; j < M; ++j) {
            const float* r_row = R + j * F;
            float acc = 0.0f;
            for (int f = 0; f < F; ++f) {
                float diff = fabsf(q_row[f] - r_row[f]);
                acc += fast_pow_host(diff, p_q16);
            }
            D[i * M + j] = fast_pow_host(acc, inv_p_q16);
        }
    }
}

/*__global__ void minkowski_kernel_fast_lut(
    const float* __restrict__ Q,
    const float* __restrict__ R,
    float* __restrict__ D,
    const int32_t* __restrict__ g_Elog,
    const int32_t* __restrict__ g_Eexp,
    int N, int M, int F,
    int32_t p_q16, int32_t inv_p_q16)
{
    // Total shared memory per block: 8 KB (LUTs) + 4 KB (Q/R tiles) = 12 KB
    __shared__ int32_t s_Elog[TABLE_SIZE];
    __shared__ int32_t s_Eexp[TABLE_SIZE];
    __shared__ float s_Q[TILE_M][TILE_F];
    __shared__ float s_R[TILE_N][TILE_F];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx; // 0..255

    // Cooperatively load LUTs into shared memory (1024 / 256 = 4 ints per thread)
    for (int idx = tid; idx < TABLE_SIZE; idx += 256) {
        s_Elog[idx] = g_Elog[idx];
        s_Eexp[idx] = g_Eexp[idx];
    }

    int q_row = blockIdx.y * TILE_M + ty;
    int r_col = blockIdx.x * TILE_N + tx;
    int r_load_row = blockIdx.x * TILE_N + ty;

    float acc = 0.0f;

    for (int f_base = 0; f_base < F; f_base += TILE_F) {
        #pragma unroll
        for (int step = 0; step < TILE_F; step += 16) {
            int f_idx = f_base + tx + step;
            s_Q[ty][tx + step] = (q_row < N && f_idx < F) ? Q[q_row * F + f_idx] : 0.0f;
            s_R[ty][tx + step] = (r_load_row < M && f_idx < F) ? R[r_load_row * F + f_idx] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_F; ++k) {
            float diff = fabsf(s_Q[ty][k] - s_R[tx][k]);
            acc += fast_pow_dev(diff, p_q16, s_Elog, s_Eexp);
        }
        __syncthreads();
    }

    if (q_row < N && r_col < M) {
        D[q_row * M + r_col] = fast_pow_dev(acc, inv_p_q16, s_Elog, s_Eexp);
    }
}*/

// ============================================================================
// Benchmark & Accuracy Harness
// ============================================================================
int main() {
    init_tables();

    // Typical ML batch dimensions (e.g., 512 queries vs 2048 reference vectors, 128 features)
    const int N = 512;
    const int M = 2048;
    const int F = 128;
    const float p = 3.0f; // L3 Minkowski norm

    const int32_t p_q16     = (int32_t)lround((double)p * 65536.0);
    const int32_t inv_p_q16 = (int32_t)lround((1.0 / (double)p) * 65536.0);

    size_t bytes_Q = (size_t)N * F * sizeof(float);
    size_t bytes_R = (size_t)M * F * sizeof(float);
    size_t bytes_D = (size_t)N * M * sizeof(float);
    size_t bytes_T = TABLE_SIZE * sizeof(int32_t);

    std::vector<float> h_Q(N * F);
    std::vector<float> h_R(M * F);
    std::vector<float> h_D_cpu_exact(N * M);
    std::vector<float> h_D_cpu_fast(N * M);
    //std::vector<float> h_D_gpu_exact(N * M);
    //std::vector<float> h_D_gpu_fast(N * M);

    srand(42);
    for (int i = 0; i < N * F; ++i) h_Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < M * F; ++i) h_R[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

    // --- CPU Benchmarks ---
    auto t0 = std::chrono::high_resolution_clock::now();
    minkowski_cpu_exact(h_Q.data(), h_R.data(), h_D_cpu_exact.data(), N, M, F, p);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_exact_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto t2 = std::chrono::high_resolution_clock::now();
    minkowski_cpu_fast_lut(h_Q.data(), h_R.data(), h_D_cpu_fast.data(), N, M, F, p);
    auto t3 = std::chrono::high_resolution_clock::now();
    double cpu_fast_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // --- Device Allocations ---
    /*float *d_Q, *d_R, *d_D_exact, *d_D_fast;
    int32_t *d_Elog, *d_Eexp;

    CUDA_CHECK(cudaMalloc(&d_Q, bytes_Q));
    CUDA_CHECK(cudaMalloc(&d_R, bytes_R));
    CUDA_CHECK(cudaMalloc(&d_D_exact, bytes_D));
    CUDA_CHECK(cudaMalloc(&d_D_fast, bytes_D));
    CUDA_CHECK(cudaMalloc(&d_Elog, bytes_T));
    CUDA_CHECK(cudaMalloc(&d_Eexp, bytes_T));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), bytes_Q, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_R, h_R.data(), bytes_R, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Elog, h_Elog, bytes_T, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Eexp, h_Eexp, bytes_T, cudaMemcpyHostToDevice));

    dim3 block(TILE_N, TILE_M);
    dim3 grid((M + TILE_N - 1) / TILE_N, (N + TILE_M - 1) / TILE_M);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warmup
    minkowski_kernel_exact<<<grid, block>>>(d_Q, d_R, d_D_exact, N, M, F, p, 1.0f / p);
    minkowski_kernel_fast_lut<<<grid, block>>>(d_Q, d_R, d_D_fast, d_Elog, d_Eexp, N, M, F, p_q16, inv_p_q16);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- GPU Exact Benchmark ---
    CUDA_CHECK(cudaEventRecord(start));
    minkowski_kernel_exact<<<grid, block>>>(d_Q, d_R, d_D_exact, N, M, F, p, 1.0f / p);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpu_exact_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_exact_ms, start, stop));

    // --- GPU Fast LUT Benchmark ---
    CUDA_CHECK(cudaEventRecord(start));
    minkowski_kernel_fast_lut<<<grid, block>>>(d_Q, d_R, d_D_fast, d_Elog, d_Eexp, N, M, F, p_q16, inv_p_q16);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float gpu_fast_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_fast_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_D_gpu_exact.data(), d_D_exact, bytes_D, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_D_gpu_fast.data(), d_D_fast, bytes_D, cudaMemcpyDeviceToHost));


    // --- Verify Accuracy ---
    double max_rel_err = 0.0;
    double mean_rel_err = 0.0;
    for (int i = 0; i < N * M; ++i) {
        double ref = (double)h_D_cpu_exact[i];
        double approx = (double)h_D_gpu_fast[i];
        double rel = (ref > 1e-9) ? fabs(approx - ref) / ref : fabs(approx - ref);
        max_rel_err = std::max(max_rel_err, rel);
        mean_rel_err += rel;
    }
    mean_rel_err /= (double)(N * M);
*/
    printf("Minkowski Distance Matrix (%d x %d, F=%d, p=%.1f, LUT K=%d)\n", N, M, F, p, K);
    printf("------------------------------------------------------------------\n");
    printf("CPU Exact (powf):     %8.3f ms\n", cpu_exact_ms);
    printf("CPU Fast  (LUT):      %8.3f ms  (%.2fx vs CPU Exact)\n",
           cpu_fast_ms, cpu_exact_ms / cpu_fast_ms);
    /*printf("T4  Exact (powf):     %8.3f ms\n", gpu_exact_ms);
    printf("T4  Fast  (LUT):      %8.3f ms  (%.2fx vs T4 Exact)\n",
           gpu_fast_ms, gpu_exact_ms / gpu_fast_ms);
    printf("Mean Relative Error:  %.6f%%\n", mean_rel_err * 100.0);
    printf("Max  Relative Error:  %.6f%%\n", max_rel_err * 100.0);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_R));
    CUDA_CHECK(cudaFree(d_D_exact));
    CUDA_CHECK(cudaFree(d_D_fast));
    CUDA_CHECK(cudaFree(d_Elog));
    CUDA_CHECK(cudaFree(d_Eexp));
*/
    return 0;
}
