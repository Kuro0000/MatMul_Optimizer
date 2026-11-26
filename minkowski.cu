// Matrice delle distanze di Minkowski su CPU e GPU, con e senza il trucco dei bit IEEE 754
//
//     C[i][j] = somma_k |A[i][k] - B[j][k]|^p        A: N query x D feature, B: M punti x D feature
//
// E' la distanza L_p elevata alla p, usata per k-NN e clustering con p frazionario (per il k-NN
// la radice 1/p finale non serve, perche' non cambia l'ordine dei vicini).
// La struttura e' quella della moltiplicazione tra matrici C = A * B^T, in cui il prodotto
// diventa |a - b|^p. Sulla GPU ogni termine costa due istruzioni della SFU (log2 ed exp2), e la
// SFU completa solo 16 operazioni per ciclo per SM: e' lei il collo di bottiglia.
//
// Quattro versioni:
//   CPU standard  powf, sequenziale: la baseline
//   CPU trucco    la potenza calcolata con il trucco dei bit
//   GPU SFU       tiling in shared memory, __powf: la potenza la calcola la SFU
//   GPU ibrido    come la SFU, ma QUOTA termini su 8 con il trucco dei bit, che lavora sulla
//                 pipeline intera: SFU e pipeline intera lavorano in parallelo
//
// Compilazione:  nvcc -O3 -arch=sm_75 -ftz=true minkowski.cu -o minkowski
//   -ftz=true (flush to zero) fa azzerare i subnormali alla GPU. Senza, __powf non costa 3
//   istruzioni ma circa 12, perche' il compilatore aggiunge i controlli sui subnormali prima di
//   ogni istruzione della SFU: e' il costo dei subnormali visto nel capitolo sul floating point.
//   Il programma avvisa se il flag manca. Non tocca la parte CPU, compilata dal compilatore host.
//   Con -DQUOTA=n si sceglie quanti termini su 8 usano il trucco (default 2, da 0 a 8).
//   Con -DPAD=0 si toglie il padding e si vede il bank conflict a 32 vie su Bs[tx][k].
// Uso:           ./minkowski [N] [M] [D] [p] [cpu] [reps]     default: 1024 1024 256 0.5 1 7
//                N, M, D multipli di 32; 0 < p < 8
//                cpu = 0 salta la CPU: il riferimento per l'errore diventa la GPU SFU
//                reps = 1 lancia ogni kernel una sola volta, per Nsight Compute:
//                ncu --set full -k regex:minkowski -o minkowski ./minkowski 1024 1024 256 0.5 0 1
// In Nsight Compute (Compute Workload Analysis): con la versione SFU la pipeline piu' usata e' la
// XU, cioe' la SFU; con l'ibrido lavorano insieme la XU e la ALU (interi). Nella tabella Shared
// Memory i bank conflict devono restare vicini a zero grazie a padding e tabelle replicate.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <limits.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                     \
{                                                                       \
    const cudaError_t error = call;                                     \
    if (error != cudaSuccess)                                           \
    {                                                                   \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                   \
        printf("code: %d, reason: %s\n", error,                         \
                cudaGetErrorString(error));                             \
        exit(1);                                                        \
    }                                                                   \
}

#define TILE 32
#ifndef PAD
#define PAD 1                       // padding della tile di B
#endif
#ifndef QUOTA
#define QUOTA 2                     // termini su 8 col trucco nella versione ibrida (2 = il migliore sulla T4)
#endif
#if QUOTA < 0 || QUOTA > 8
#error "QUOTA deve essere tra 0 e 8"
#endif
#define K 7                         // bit delle tabelle di correzione: 2^K = 128 voci
#define NT (1 << K)
#define BIAS_32 ((int32_t)127 << 23)
#define LP_MIN (-126 * (1 << 23))   // p * log2|d| limitato a -126: sotto, |d|^p e' ~0
#define REPS_MAX 32

double cpuSecond() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ((double)ts.tv_sec + (double)ts.tv_nsec * 1.e-9);
}

// ---------------------------------------------------------------------------
// Tabelle di correzione (centro della cella, come nei vostri file), con il BIAS gia' incluso:
//   log2:  L = bit(|d|) + tab_log[r] = (bit(|d|) - BIAS) + Elog[r]     -> log2|d| * 2^23
//   exp2:  bit(risultato) = Lp + tab_exp[r] = Lp + BIAS + Eexp[r]
// ---------------------------------------------------------------------------
static int32_t h_log[NT], h_exp[NT];
__constant__ int32_t c_log[NT];
__constant__ int32_t c_exp[NT];

static void init_tabelle(void) {
    for (int i = 0; i < NT; i++) {
        double x = (i + 0.5) / NT;
        h_log[i] = (int32_t)lround((log2(1 + x) - x) * (1 << 23)) - BIAS_32;
        h_exp[i] = (int32_t)lround((exp2(x) - 1 - x) * (1 << 23)) + BIAS_32;
    }
    CHECK(cudaMemcpyToSymbol(c_log, h_log, sizeof h_log));
    CHECK(cudaMemcpyToSymbol(c_exp, h_exp, sizeof h_exp));
}

// Cella della tabella: i K bit piu' alti della mantissa (o della parte frazionaria del log)
#define RIGA(x) (((x) >> (23 - K)) & (NT - 1))

// ---------------------------------------------------------------------------
// CPU
// ---------------------------------------------------------------------------
static void minkowski_cpu(const float *A, const float *B, float *C, int N, int M, int D, float p) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) {
            float acc = 0.0f;
            for (int k = 0; k < D; k++)
                acc += powf(fabsf(A[i * D + k] - B[j * D + k]), p);
            C[i * M + j] = acc;
        }
}

// |d|^p con il trucco dei bit, in tre passi:
//  1. log2|d|: i bit del float letti come intero, meno il BIAS, piu' la correzione (tabella log)
//  2. per p: moltiplicazione in virgola fissa. Con P = p * 2^28, i 32 bit alti del prodotto a
//     64 bit L * P valgono p * L / 16, quindi si moltiplica per 16
//  3. exp2: l'intero torna a essere i bit di un float, piu' la correzione (tabella exp)
// Valida per 0 < p < 8 e |d|^p rappresentabile in float; con |d| = 0 il risultato e' ~0
// (al piu' 2^-126), grazie al limite LP_MIN.
static inline float pow_trucco_cpu(float d, int32_t P) {
    int32_t b;
    memcpy(&b, &d, sizeof b);
    b &= 0x7FFFFFFF;                                              // bit di |d|
    int32_t L = b + h_log[RIGA(b)];                               // log2|d| in virgola fissa
    int32_t q = (int32_t)(((int64_t)L * P) >> 32);                // p * log2|d| / 16
    if (q < LP_MIN / 16) q = LP_MIN / 16;
    int32_t Lp = q * 16;
    int32_t r = Lp + h_exp[RIGA(Lp)];                             // exp2: di nuovo i bit di un float
    float y;
    memcpy(&y, &r, sizeof y);
    return y;
}

static void minkowski_cpu_trucco(const float *A, const float *B, float *C, int N, int M, int D, int32_t P) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++) {
            float acc = 0.0f;
            for (int k = 0; k < D; k++)
                acc += pow_trucco_cpu(A[i * D + k] - B[j * D + k], P);
            C[i * M + j] = acc;
        }
}

// ---------------------------------------------------------------------------
// GPU SFU: tiling in shared memory, come nella moltiplicazione tra matrici a tile.
// Ogni blocco calcola una tile 32x32 di C. A ogni passo carica 32 feature di 32 righe di A e di
// 32 righe di B con letture coalescenti (tx scorre le feature) e le riusa 32 volte.
// Nel ciclo interno As[ty][k] e' uguale per tutto il warp (broadcast), mentre Bs[tx][k] e' letto
// per colonne: con Bs[32][32] i 32 thread cadrebbero nello stesso banco (conflitto a 32 vie),
// con il padding Bs[32][33] ogni riga e' sfalsata di una word e i banchi sono tutti diversi.
// __powf calcola |d|^p = exp2(p * log2|d|) con due istruzioni della SFU (MUFU.LG2, MUFU.EX2)
// e una moltiplicazione.
// __launch_bounds__(1024): con blocchi da 1024 thread il compilatore deve stare entro 64 registri
// per thread (65.536 in tutto, quelli di un SM della T4), altrimenti il kernel non parte.
// Richiede N, M e D multipli di TILE.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE * TILE) minkowski_sfu(const float *A, const float *B, float *C, int N, int M, int D, float p) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE + PAD];
    int tx = threadIdx.x, ty = threadIdx.y;
    int i = blockIdx.y * TILE + ty;      // query di questo thread
    int j = blockIdx.x * TILE + tx;      // punto di questo thread
    int jb = blockIdx.x * TILE + ty;     // riga di B che questo thread carica nella tile
    float acc = 0.0f;

    for (int k0 = 0; k0 < D; k0 += TILE) {
        As[ty][tx] = A[i * D + k0 + tx];
        Bs[ty][tx] = B[jb * D + k0 + tx];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++)
            acc += __powf(fabsf(As[ty][k] - Bs[tx][k]), p);
        __syncthreads();
    }
    C[i * M + j] = acc;
}

// ---------------------------------------------------------------------------
// |d|^p con il trucco dei bit sulla GPU: stessa aritmetica della CPU (__mulhi da' i 32 bit alti
// del prodotto a 64 bit, come ((int64_t)L * P) >> 32), quindi stessi risultati.
// Le tabelle sono in shared memory, replicate: la copia della riga r per il lane l sta nella
// word r * 32 + l, cioe' nel banco l. Ogni thread legge sempre dal proprio banco, quindi nessun
// bank conflict, qualunque siano i dati. (RIGA(x) << 5) | lane e' r * 32 + l scritto in modo
// che il compilatore usi meno istruzioni.
// Non si converte mai l'intero in float: sulla GPU le conversioni le fa la stessa unita' della
// SFU, che vogliamo scaricare.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float pow_trucco(float d, int32_t P, const int32_t *sLog, const int32_t *sExp, int lane) {
    int32_t b = __float_as_int(d) & 0x7FFFFFFF;
    int32_t L = b + sLog[(RIGA(b) << 5) | lane];
    int32_t Lp = max(__mulhi(L, P), LP_MIN / 16) * 16;
    return __int_as_float(Lp + sExp[(RIGA(Lp) << 5) | lane]);
}

// ---------------------------------------------------------------------------
// GPU ibrido: come la versione SFU, ma dei 32 termini di ogni passo QUOTA su 8 usano il trucco
// (pipeline intera) e gli altri __powf (SFU), cosi' le due unita' lavorano in parallelo.
// k e' noto al compilatore dopo l'unrolling, quindi la scelta del ramo e' gia' fatta in
// compilazione, e tutti i thread del warp seguono lo stesso ramo: nessuna divergenza.
// Le tabelle replicate occupano 2 x 16 KB. Con blockDim.x = 32 il warp e' una riga del blocco
// e tx coincide con il lane. Ogni warp copia le righe ty, ty + 32, ...: tutti i suoi thread
// leggono lo stesso c_log[r] (broadcast della constant memory) e scrivono 32 word consecutive.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE * TILE) minkowski_ibrido(const float *A, const float *B, float *C, int N, int M, int D, float p, int32_t P) {
    __shared__ int32_t sLog[NT * 32], sExp[NT * 32];
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE + PAD];
    int tx = threadIdx.x, ty = threadIdx.y;
    for (int r = ty; r < NT; r += TILE) {
        sLog[r * 32 + tx] = c_log[r];
        sExp[r * 32 + tx] = c_exp[r];
    }
    __syncthreads();

    int i = blockIdx.y * TILE + ty;
    int j = blockIdx.x * TILE + tx;
    int jb = blockIdx.x * TILE + ty;
    float acc = 0.0f;

    for (int k0 = 0; k0 < D; k0 += TILE) {
        As[ty][tx] = A[i * D + k0 + tx];
        Bs[ty][tx] = B[jb * D + k0 + tx];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            float d = As[ty][k] - Bs[tx][k];
            if (k % 8 < QUOTA)
                acc += pow_trucco(d, P, sLog, sExp, tx);   // pipeline intera
            else
                acc += __powf(fabsf(d), p);                 // SFU
        }
        __syncthreads();
    }
    C[i * M + j] = acc;
}

// Controllo del flag -ftz=true: con ftz il prodotto FLT_MIN * 0.5, che e' subnormale, diventa 0
__global__ void prova_ftz(float x, float *y) {
    *y = x * 0.5f;
}

// ---------------------------------------------------------------------------
// Misura e accuratezza
// ---------------------------------------------------------------------------
static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void lancia(int ibrido, dim3 grid, dim3 block, const float *dA, const float *dB, float *dC,
                   int N, int M, int D, float p, int32_t P) {
    if (ibrido)
        minkowski_ibrido<<<grid, block>>>(dA, dB, dC, N, M, D, p, P);
    else
        minkowski_sfu<<<grid, block>>>(dA, dB, dC, N, M, D, p);
    CHECK(cudaGetLastError());
}

// Riscaldamento: a riposo la T4 abbassa il clock e lo rialza solo dopo un po' di lavoro
// continuo. Senza riscaldamento, kernel da pochi millisecondi vengono misurati a frequenza
// ridotta, e i tempi dipendono da quando capita la misura.
static void riscalda(dim3 grid, dim3 block, const float *dA, const float *dB, float *dC,
                     int N, int M, int D, float p, int32_t P) {
    double t0 = cpuSecond();
    while (cpuSecond() - t0 < 0.5) {
        for (int r = 0; r < 10; r++)                  // 10 lanci in coda: la GPU non si ferma
            lancia(r % 2, grid, block, dA, dB, dC, N, M, D, p, P);
        CHECK(cudaDeviceSynchronize());
    }
}

// Misura alternata: SFU, ibrido, SFU, ibrido, ... Se il clock cambia durante le misure, cambia
// per entrambe le versioni allo stesso modo. Restituisce la mediana di ciascuna.
static void misura(int reps, dim3 grid, dim3 block, const float *dA, const float *dB,
                   float *dC_sfu, float *dC_ibr, int N, int M, int D, float p, int32_t P,
                   double *t_sfu, double *t_ibr) {
    double t[2][REPS_MAX];
    for (int r = 0; r < reps; r++)
        for (int v = 0; v < 2; v++) {
            double t0 = cpuSecond();
            lancia(v, grid, block, dA, dB, v ? dC_ibr : dC_sfu, N, M, D, p, P);
            CHECK(cudaDeviceSynchronize());
            t[v][r] = cpuSecond() - t0;
        }
    qsort(t[0], reps, sizeof(double), cmp_double);
    qsort(t[1], reps, sizeof(double), cmp_double);
    *t_sfu = t[0][reps / 2];
    *t_ibr = t[1][reps / 2];
}

// Per ogni query, l'indice del punto piu' vicino (1-NN)
static void vicini(const float *C, int N, int M, int *nn) {
    for (int i = 0; i < N; i++) {
        int best = 0;
        for (int j = 1; j < M; j++)
            if (C[(size_t)i * M + j] < C[(size_t)i * M + best]) best = j;
        nn[i] = best;
    }
}

// Errore relativo massimo rispetto al riferimento e percentuale di query con lo stesso vicino
static void accuratezza(const float *C, const float *ref, const int *nn_ref, int *nn, int N, int M,
                        double *emax, double *uguali) {
    double e = 0;
    for (size_t x = 0; x < (size_t)N * M; x++) {
        double diff = fabs((double)C[x] - ref[x]);
        double r = (ref[x] != 0.0f) ? diff / ref[x] : diff;   // punti identici: distanza 0
        if (r > e) e = r;
    }
    vicini(C, N, M, nn);
    int u = 0;
    for (int i = 0; i < N; i++) u += (nn[i] == nn_ref[i]);
    *emax = e;
    *uguali = 100.0 * u / N;
}

static void riga(const char *nome, double t, double t_base, double termini, double emax, double uguali) {
    char vs[16] = "-";
    if (t_base > 0) snprintf(vs, sizeof vs, "%.1fx", t_base / t);
    printf("%-26s %11.3f %11.2f %16s %14.2e %10.1f%%\n", nome, t * 1e3, termini / t / 1e9, vs, emax, uguali);
}

int main(int argc, char **argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    int M = (argc > 2) ? atoi(argv[2]) : 1024;
    int D = (argc > 3) ? atoi(argv[3]) : 256;
    float p = (argc > 4) ? (float)atof(argv[4]) : 0.5f;
    int usa_cpu = (argc > 5) ? atoi(argv[5]) : 1;
    int reps = (argc > 6) ? atoi(argv[6]) : 7;

    if (N <= 0 || M <= 0 || D <= 0 || N % TILE || M % TILE || D % TILE) {
        printf("N, M e D devono essere positivi e multipli di %d.\n", TILE);
        return 1;
    }
    if (!(p > 0.0f && p < 8.0f)) {
        printf("Serve 0 < p < 8.\n");
        return 1;
    }
    if ((double)N * M > INT_MAX || (double)N * D > INT_MAX || (double)M * D > INT_MAX) {
        printf("Matrici troppo grandi per indici a 32 bit.\n");
        return 1;
    }
    if (reps < 1 || reps > REPS_MAX) reps = 7;

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    double termini = (double)N * M * D;
    printf("GPU: %s (%d SM, cc %d.%d)\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    printf("N = %d query, M = %d punti, D = %d feature, p = %g: %.1f milioni di termini |a-b|^p\n",
           N, M, D, p, termini / 1e6);
    printf("Versione ibrida: %d termini su 8 con il trucco\n\n", QUOTA);
    init_tabelle();
    int32_t P = (int32_t)lround(p * 268435456.0);   // p * 2^28

    // Il flag -ftz=true c'e'?
    float *dY, y;
    CHECK(cudaMalloc((void **)&dY, sizeof(float)));
    prova_ftz<<<1, 1>>>(FLT_MIN, dY);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(&y, dY, sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dY));
    if (y != 0.0f)
        printf("ATTENZIONE: compilato senza -ftz=true, __powf costa circa 12 istruzioni invece di 3.\n\n");

    size_t nA = (size_t)N * D, nB = (size_t)M * D, nC = (size_t)N * M;
    float *A = (float *)malloc(nA * sizeof(float));
    float *B = (float *)malloc(nB * sizeof(float));
    float *C_sfu = (float *)malloc(nC * sizeof(float));
    float *C_ibr = (float *)malloc(nC * sizeof(float));
    float *C_cpu = (float *)malloc(nC * sizeof(float));
    float *C_trc = (float *)malloc(nC * sizeof(float));
    int *nn_ref = (int *)malloc(N * sizeof(int)), *nn = (int *)malloc(N * sizeof(int));
    if (!A || !B || !C_sfu || !C_ibr || !C_cpu || !C_trc || !nn_ref || !nn) {
        printf("Memoria host insufficiente.\n");
        return 1;
    }
    srand(42);
    for (size_t x = 0; x < nA; x++) A[x] = (float)rand() / ((float)RAND_MAX + 1.0f);   // feature in [0, 1]
    for (size_t x = 0; x < nB; x++) B[x] = (float)rand() / ((float)RAND_MAX + 1.0f);

    // ---------------- CPU ----------------
    double t_cpu = 0, t_trc = 0;
    if (usa_cpu) {
        double t0 = cpuSecond();
        minkowski_cpu(A, B, C_cpu, N, M, D, p);
        t_cpu = cpuSecond() - t0;
        t0 = cpuSecond();
        minkowski_cpu_trucco(A, B, C_trc, N, M, D, P);
        t_trc = cpuSecond() - t0;
    }

    // ---------------- GPU ----------------
    float *dA, *dB, *dC_sfu, *dC_ibr;
    CHECK(cudaMalloc((void **)&dA, nA * sizeof(float)));
    CHECK(cudaMalloc((void **)&dB, nB * sizeof(float)));
    CHECK(cudaMalloc((void **)&dC_sfu, nC * sizeof(float)));
    CHECK(cudaMalloc((void **)&dC_ibr, nC * sizeof(float)));
    CHECK(cudaMemcpy(dA, A, nA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, B, nB * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(dC_sfu, 0, nC * sizeof(float)));
    CHECK(cudaMemset(dC_ibr, 0, nC * sizeof(float)));

    dim3 block(TILE, TILE);
    dim3 grid(M / TILE, N / TILE);

    if (reps > 1)                                      // con reps = 1 (Nsight Compute) non serve
        riscalda(grid, block, dA, dB, dC_sfu, N, M, D, p, P);
    double t_sfu, t_ibr;
    misura(reps, grid, block, dA, dB, dC_sfu, dC_ibr, N, M, D, p, P, &t_sfu, &t_ibr);
    CHECK(cudaMemcpy(C_sfu, dC_sfu, nC * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(C_ibr, dC_ibr, nC * sizeof(float), cudaMemcpyDeviceToHost));

    // ---------------- Risultati ----------------
    // Riferimento: la CPU standard; senza CPU, la GPU SFU (errore ~1e-6 rispetto a powf)
    const float *ref = usa_cpu ? C_cpu : C_sfu;
    vicini(ref, N, M, nn_ref);
    double e_trc = 0, u_trc = 0, e_sfu, u_sfu, e_ibr, u_ibr;
    if (usa_cpu) accuratezza(C_trc, ref, nn_ref, nn, N, M, &e_trc, &u_trc);
    accuratezza(C_sfu, ref, nn_ref, nn, N, M, &e_sfu, &u_sfu);
    accuratezza(C_ibr, ref, nn_ref, nn, N, M, &e_ibr, &u_ibr);

    printf("%-26s %11s %11s %16s %14s %11s\n", "versione", "tempo [ms]", "Gtermini/s",
           "vs CPU standard", "err. rel. max", "stesso NN");
    if (usa_cpu) {
        riga("CPU standard (powf)", t_cpu, t_cpu, termini, 0.0, 100.0);
        riga("CPU trucco", t_trc, t_cpu, termini, e_trc, u_trc);
    }
    riga("GPU SFU (__powf)", t_sfu, t_cpu, termini, e_sfu, u_sfu);
    char nome[32];
    snprintf(nome, sizeof nome, "GPU ibrido (%d/8 trucco)", QUOTA);
    riga(nome, t_ibr, t_cpu, termini, e_ibr, u_ibr);

    printf("\n");
    if (usa_cpu)
        printf("Trucco su CPU:   %.2fx rispetto a powf\n", t_cpu / t_trc);
    printf("Ibrido su GPU:   %.2fx rispetto alla sola SFU\n", t_sfu / t_ibr);
    printf("\nTempi GPU: mediana di %d esecuzioni alternate%s.\n", reps,
           reps > 1 ? ", dopo 0,5 s di riscaldamento" : "");
    printf("Errore rispetto a %s.\n", usa_cpu ? "CPU standard" : "GPU SFU");
    printf("stesso NN = query per cui il punto piu' vicino coincide con quello del riferimento.\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC_sfu); cudaFree(dC_ibr);
    free(A); free(B); free(C_sfu); free(C_ibr); free(C_cpu); free(C_trc); free(nn_ref); free(nn);
    return 0;
}
