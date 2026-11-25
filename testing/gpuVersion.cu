// Moltiplicazione di matrici NxN: CPU (standard e logaritmica) vs GPU CUDA (standard e logaritmica)
//
// Compilazione:  nvcc -O3 -arch=native lns_cuda.cu -o lns_cuda -lm
// Uso:           ./lns_cuda [N] [minimo] [massimo]      (default: 1024 1 1000)
//                N deve essere multiplo di 32
//
// Attenzione: il tempo CPU cresce come N^3. Con N=4096 la CPU impiega minuti.

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define BIAS_32 ((int32_t)127 << 23)
#define K 8
#define TILE 32

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err_));                                  \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Tabelle di correzione (host + copia in constant memory del device)
// ---------------------------------------------------------------------------
static int32_t Elog[1 << K], Eexp[1 << K];
__constant__ int32_t c_Elog[1 << K];
__constant__ int32_t c_Eexp[1 << K];

static void init_tables(void) {
    for (int i = 0; i < (1 << K); i++) {
        double x = (i + 0.5) / (1 << K);   // centro della cella
        Elog[i] = (int32_t)lround((log2(1 + x) - x) * (1 << 23));
        Eexp[i] = (int32_t)lround((exp2(x) - 1 - x) * (1 << 23));
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_Elog, Elog, sizeof Elog));
    CUDA_CHECK(cudaMemcpyToSymbol(c_Eexp, Eexp, sizeof Eexp));
}

// ---------------------------------------------------------------------------
// Funzioni log2/exp2 approssimate: versione host
// ---------------------------------------------------------------------------
static inline int32_t fast_log2(float x) {
    int32_t i; memcpy(&i, &x, sizeof i);
    return (i - BIAS_32) + Elog[(i >> (23 - K)) & ((1 << K) - 1)];
}

static inline float fast_exp2(int32_t L) {
    int32_t i = L + BIAS_32 + Eexp[(L >> (23 - K)) & ((1 << K) - 1)];
    float y; memcpy(&y, &i, sizeof y);
    return y;
}

// ---------------------------------------------------------------------------
// Funzioni log2/exp2 approssimate: versione device (tabella passata in shared memory)
// ---------------------------------------------------------------------------
__device__ __forceinline__ int32_t fast_log2_d(float x, const int32_t *tl) {
    int32_t i = __float_as_int(x);
    return (i - BIAS_32) + tl[(i >> (23 - K)) & ((1 << K) - 1)];
}

__device__ __forceinline__ float fast_exp2_d(int32_t L, const int32_t *te) {
    int32_t i = L + BIAS_32 + te[(L >> (23 - K)) & ((1 << K) - 1)];
    return __int_as_float(i);
}

// ---------------------------------------------------------------------------
// Kernel 1: moltiplicazione standard con tiling in shared memory
// Richiede N multiplo di TILE: niente controlli sui bordi, cicli a conteggio fisso
// ---------------------------------------------------------------------------
__global__ void matmul_std(const float *A, const float *B, float *C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    for (int t = 0; t < n / TILE; t++) {
        As[ty][tx] = A[(size_t)row * n + t * TILE + tx];
        Bs[ty][tx] = B[(size_t)(t * TILE + ty) * n + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++)
            acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    C[(size_t)row * n + col] = acc;
}

// ---------------------------------------------------------------------------
// Kernel 2: moltiplicazione logaritmica fusa
// Il log2 e' calcolato mentre si carica il tile in shared memory, poi il prodotto
// e' exp2(logA + logB): solo somme intere nel ciclo interno
// Richiede N multiplo di TILE
// ---------------------------------------------------------------------------
static_assert(TILE * TILE >= (1 << K), "servono almeno 2^K thread per blocco per caricare le tabelle");

__global__ void matmul_log(const float *A, const float *B, float *C, int n) {
    __shared__ int32_t sLog[1 << K];
    __shared__ int32_t sExp[1 << K];
    __shared__ int32_t As[TILE][TILE];
    __shared__ int32_t Bs[TILE][TILE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int tid = ty * TILE + tx;
    if (tid < (1 << K)) {          // un solo passaggio: nessun ciclo
        sLog[tid] = c_Elog[tid];
        sExp[tid] = c_Eexp[tid];
    }
    __syncthreads();               // le tabelle servono gia' durante il caricamento dei tile

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;
    float acc = 0.0f;

    for (int t = 0; t < n / TILE; t++) {
        As[ty][tx] = fast_log2_d(A[(size_t)row * n + t * TILE + tx], sLog);
        Bs[ty][tx] = fast_log2_d(B[(size_t)(t * TILE + ty) * n + col], sLog);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++)
            acc += fast_exp2_d(As[ty][k] + Bs[k][tx], sExp);
        __syncthreads();
    }
    C[(size_t)row * n + col] = acc;
}

// ---------------------------------------------------------------------------
// Utilita'
// ---------------------------------------------------------------------------
static float casuale(double minimo, double massimo) {
    return (float)(minimo + (massimo - minimo) * ((double)rand() / RAND_MAX));
}

// Errore relativo di 'test' rispetto a 'ref' su tutta la matrice
static void confronta(const char *nome, const float *test, const float *ref, size_t cnt, double tol) {
    double emax = 0, esomma = 0, esigned = 0;
    for (size_t i = 0; i < cnt; i++) {
        double e = ((double)test[i] - ref[i]) / ref[i];
        esigned += e;
        double a = fabs(e);
        esomma += a;
        if (a > emax) emax = a;
    }
    printf("%-28s max %.6f%%   medio(|e|) %.6f%%   medio(con segno) %+.6f%%   %s\n",
           nome, 100 * emax, 100 * esomma / cnt, 100 * esigned / cnt,
           (tol > 0) ? (emax <= tol ? "[OK]" : "[ERRORE]") : "[approssimazione]");
}

static double gflops(int n, double secondi) {
    return secondi > 0 ? 2.0 * n * n * (double)n / secondi / 1e9 : 0;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 1024;
    double minimo = (argc > 2) ? atof(argv[2]) : 1.0;
    double massimo = (argc > 3) ? atof(argv[3]) : 1000.0;

    if (n <= 0 || n % TILE != 0 || minimo <= 0 || massimo <= minimo) {
        fprintf(stderr, "Servono N>0 multiplo di %d e due numeri positivi con massimo > minimo.\n", TILE);
        return 1;
    }

    size_t cnt = (size_t)n * n;
    size_t bytes = cnt * sizeof(float);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | N = %d | range [%g, %g]\n\n", prop.name, n, minimo, massimo);
    CUDA_CHECK(cudaFree(0));   // inizializza il contesto CUDA fuori dalle misure
    init_tables();

    // ---------------- Host ----------------
    float *A = (float *)malloc(bytes), *B = (float *)malloc(bytes);
    float *C_cpu_std = (float *)calloc(cnt, sizeof(float));
    float *C_cpu_log = (float *)calloc(cnt, sizeof(float));
    float *C_gpu_std = (float *)malloc(bytes), *C_gpu_log = (float *)malloc(bytes);
    if (!A || !B || !C_cpu_std || !C_cpu_log || !C_gpu_std || !C_gpu_log) {
        fprintf(stderr, "Memoria host insufficiente.\n");
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < cnt; i++) {
        A[i] = casuale(minimo, massimo);
        B[i] = casuale(minimo, massimo);
    }

    // ---------------- CPU standard ----------------
    clock_t t0 = clock();
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            float a = A[(size_t)i * n + k];
            for (int j = 0; j < n; j++)
                C_cpu_std[(size_t)i * n + j] += a * B[(size_t)k * n + j];
        }
    double t_cpu_std = (double)(clock() - t0) / CLOCKS_PER_SEC;

    // ---------------- CPU logaritmica ----------------
    t0 = clock();
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            int32_t log_a = fast_log2(A[(size_t)i * n + k]);
            for (int j = 0; j < n; j++)
                C_cpu_log[(size_t)i * n + j] += fast_exp2(log_a + fast_log2(B[(size_t)k * n + j]));
        }
    double t_cpu_log = (double)(clock() - t0) / CLOCKS_PER_SEC;

    // ---------------- Device ----------------
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    float ms;

    // Host -> Device
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    double t_h2d = ms / 1000.0;

    dim3 block(TILE, TILE);
    dim3 grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    // Kernel standard
    CUDA_CHECK(cudaMemset(dC, 0, bytes));
    CUDA_CHECK(cudaEventRecord(e0));
    matmul_std<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    double t_gpu_std = ms / 1000.0;
    CUDA_CHECK(cudaMemcpy(C_gpu_std, dC, bytes, cudaMemcpyDeviceToHost));

    // Kernel logaritmico fuso (conversione in log + prodotto)
    CUDA_CHECK(cudaMemset(dC, 0, bytes));
    CUDA_CHECK(cudaEventRecord(e0));
    matmul_log<<<grid, block>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    double t_gpu_log = ms / 1000.0;

    // Device -> Host
    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(C_gpu_log, dC, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    double t_d2h = ms / 1000.0;

    // ---------------- Tempi ----------------
    printf("=== TEMPI ===\n");
    printf("CPU standard:                  %10.4f s   (%8.2f GFLOPS)\n", t_cpu_std, gflops(n, t_cpu_std));
    printf("CPU logaritmica:               %10.4f s   (%8.2f GFLOPS)\n", t_cpu_log, gflops(n, t_cpu_log));
    printf("GPU standard (solo kernel):    %10.4f s   (%8.2f GFLOPS)\n", t_gpu_std, gflops(n, t_gpu_std));
    printf("GPU logaritmica (kernel fuso): %10.4f s   (%8.2f GFLOPS equiv.)\n", t_gpu_log, gflops(n, t_gpu_log));
    printf("Trasferimento H->D (A e B):    %10.4f s\n", t_h2d);
    printf("Trasferimento D->H (C):        %10.4f s\n", t_d2h);

    printf("\n=== SPEEDUP ===\n");
    printf("CPU std / CPU log:             %.2fx\n", t_cpu_std / t_cpu_log);
    printf("CPU std / GPU std:             %.1fx\n", t_cpu_std / t_gpu_std);
    printf("CPU log / GPU log:             %.1fx\n", t_cpu_log / t_gpu_log);
    printf("GPU std / GPU log:             %.2fx\n", t_gpu_std / t_gpu_log);

    // ---------------- Correttezza ----------------
    // GPU vs CPU sulla stessa operazione: differiscono solo per l'ordine di somma (tolleranza 1e-4)
    // log vs standard: errore intrinseco dell'approssimazione logaritmica
    printf("\n=== ERRORE RELATIVO SULLA MATRICE ===\n");
    confronta("GPU std  vs CPU std",  C_gpu_std, C_cpu_std, cnt, 1e-4);
    confronta("GPU log  vs CPU log",  C_gpu_log, C_cpu_log, cnt, 1e-4);
    confronta("CPU log  vs CPU std",  C_cpu_log, C_cpu_std, cnt, 0);
    confronta("GPU log  vs CPU std",  C_gpu_log, C_cpu_std, cnt, 0);
    confronta("GPU log  vs GPU std",  C_gpu_log, C_gpu_std, cnt, 0);

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C_cpu_std); free(C_cpu_log); free(C_gpu_std); free(C_gpu_log);
    return 0;
}