// Esponenziale 2^x su GPU: unita' hardware (MUFU) contro il trucco dei bit IEEE 754
//
// E' il caso d'uso di FlashAttention-4: nella softmax dell'attention il collo di bottiglia sono
// gli esponenziali, perche' la MUFU (l'unita' delle funzioni speciali) completa solo 16 operazioni
// per ciclo per SM, mentre sulla T4 la pipeline FMA e quella intera ne completano 64 ciascuna.
// Se una parte degli exp2 viene calcolata con il trucco dei bit su queste altre pipeline,
// le due strade lavorano in parallelo e il throughput totale sale.
//
// Metodi confrontati:
//  - MUFU:      ex2.approx.ftz.f32, cioe' una sola istruzione MUFU.EX2.
//  - tabella:   il vostro trucco. x in virgola fissa viene scritto direttamente nei bit del float
//               (parte intera -> esponente, parte frazionaria -> mantissa, cioe' 2^f ~ 1 + f) e
//               corretto con una tabella da 256 voci, replicata per banco come in matmul_log_rep.
//               Lavora quasi solo sulla pipeline intera, che nella softmax e' ferma.
//  - polinomio: come FlashAttention-4. x = n + f, 2^f con una cubica sulle FMA, poi n sommato
//               direttamente al campo esponente. Lavora sulla pipeline FMA.
// Le righe "k/8" calcolano k esponenziali su 8 con il metodo software e gli altri con la MUFU.
//
// Il kernel e' una softmax sintetica: ogni thread tiene 8 punteggi in registro e a ogni passo
// calcola 2^(s*log2(e) - m) e accumula la somma, come nel ciclo interno dell'attention.
// I dati stanno nei registri apposta: cosi' il limite e' il calcolo, non la memoria.
// Da sola, una softmax che legge dalla memoria globale e' limitata dalla banda e il trucco
// non si vedrebbe.
//
// Compilazione:  nvcc -O3 -std=c++17 -arch=native exp2_trick.cu -o exp2_trick
// Uso:           ./exp2_trick

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err_));                                  \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define FLUSSI 8            // esponenziali indipendenti per thread a ogni passo
#define PASSI 4096          // passi del ciclo per thread
#define BLOCCO 1024         // thread per blocco
#define BLOCCHI_PER_SM 8
#define REPS 7              // esecuzioni misurate per configurazione (si riporta la mediana)

#define K 8                 // bit della tabella di correzione: 2^K = 256 voci
#define MAGIC 12582912.0f   // 1.5 * 2^23: sommato a un float lascia nei bit bassi l'intero arrotondato

// Cubica minimax (errore relativo) per 2^f su [-0.5, 0.5] con p(0) = 1: errore massimo ~1e-4
#define P1 0.693282902f
#define P2 0.242210627f
#define P3 0.0550086908f

enum { MUFU = 0, TAB = 1, POLY = 2 };

// Tabella di correzione di exp2 (come Eexp nei vostri file), con gia' sommate le costanti
// che servono a exp2_tab: valore = Eexp[i] - 2^23 (vedi sotto)
__constant__ int32_t c_tab[1 << K];

// ---------------------------------------------------------------------------
// 2^x hardware: una sola istruzione MUFU.EX2
// ---------------------------------------------------------------------------
__device__ __forceinline__ float exp2_hw(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// ---------------------------------------------------------------------------
// 2^x con il trucco dei bit e la tabella di correzione
//
// Il chiamante passa t = MAGIC + x * 2^15, che costa la stessa FFMA che serve comunque a
// formare x. Per come e' fatto MAGIC, i bit bassi di t contengono round(x * 2^15), cioe' x in
// virgola fissa con 15 bit frazionari, senza nessuna conversione float -> int (che sulla GPU
// e' lenta quanto la MUFU).
// (t << 8) porta x a 23 bit frazionari: la parte intera finisce nel campo esponente e la parte
// frazionaria nella mantissa. In piu' contiene 2^30, il residuo di MAGIC: per questo la tabella
// contiene Eexp - 2^23 = Eexp + BIAS - 2^30.
// L'indice sono gli 8 bit frazionari alti (bit 7..14 di t), e ogni lane legge la propria copia
// della tabella, che sta nel banco 'lane': nessun bank conflict.
// Valido per -126 < x < 128.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float exp2_tab(float t, const int32_t *tab, uint32_t lane) {
    uint32_t b = __float_as_uint(t);
    uint32_t idx = ((b >> (10 - K)) & (((1u << K) - 1) << 5)) | lane;   // riga * 32 + lane
    return __uint_as_float((b << 8) + (uint32_t)tab[idx]);
}

// ---------------------------------------------------------------------------
// 2^x come in FlashAttention-4: x = n + f con f in [-0.5, 0.5], 2^f con una cubica sulle FMA,
// poi n sommato direttamente al campo esponente (n sta nei bit bassi di t, quindi t << 23).
// __fadd_rn/__fsub_rn impediscono al compilatore di fondere o riordinare le somme.
// Valido per -126 < x < 127.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float exp2_poly(float x) {
    float t = __fadd_rn(x, MAGIC);
    float f = __fsub_rn(x, __fsub_rn(t, MAGIC));
    float p = fmaf(fmaf(fmaf(P3, f, P2), f, P1), f, 1.0f);
    return __uint_as_float(__float_as_uint(p) + (__float_as_uint(t) << 23));
}

// ---------------------------------------------------------------------------
// Indice del thread nel warp, letto dal registro speciale %laneid. Con threadIdx.x & 31 il
// compilatore rifarebbe l'AND dentro ogni calcolo dell'indice: un'istruzione in piu' per exp2.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t lane_id(void) {
    uint32_t l;
    asm("mov.u32 %0, %%laneid;" : "=r"(l));
    return l;
}

// ---------------------------------------------------------------------------
// Tabella replicata in shared memory: riga r, copia del lane l nella parola r * 32 + l (banco l).
// Ogni warp copia righe intere: tutti i lane leggono lo stesso c_tab[r] (broadcast).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void carica_tabella(int32_t *s_tab) {
    uint32_t lane = threadIdx.x & 31;
    for (int r = threadIdx.x >> 5; r < (1 << K); r += blockDim.x / 32)
        s_tab[r * 32 + lane] = c_tab[r];
    __syncthreads();
}

// ---------------------------------------------------------------------------
// Softmax sintetica: E esponenziali su FLUSSI con il metodo software, gli altri con la MUFU.
// METODO ed E sono parametri del template, quindi la scelta non costa nessun salto.
// ---------------------------------------------------------------------------
template <int METODO, int E>
__global__ void __launch_bounds__(BLOCCO) softmax_exp2(const float *S, float *out, float c, float dm) {
    const uint32_t lane = lane_id();
    const int gid = blockIdx.x * BLOCCO + threadIdx.x;
    const int32_t *tab = nullptr;
    if constexpr (METODO == TAB) {
        __shared__ int32_t s_tab[(1 << K) * 32];   // 32 KB
        carica_tabella(s_tab);
        tab = s_tab;
    }

    float s[FLUSSI], acc[FLUSSI];
    #pragma unroll
    for (int u = 0; u < FLUSSI; u++) {
        s[u] = S[(size_t)gid * FLUSSI + u];
        acc[u] = 0.0f;
    }
    const float c15 = c * 32768.0f;   // per la tabella: t = s*c*2^15 + (MAGIC - m*2^15) = MAGIC + x*2^15
    float m = 0.0f;                   // "massimo corrente" della softmax (qui cresce in modo regolare)

    #pragma unroll 4
    for (int p = 0; p < PASSI; p++) {
        const float km = fmaf(m, -32768.0f, MAGIC);
        #pragma unroll
        for (int u = 0; u < FLUSSI; u++) {
            float y;
            if (METODO == TAB && u < E)
                y = exp2_tab(fmaf(s[u], c15, km), tab, lane);
            else if (METODO == POLY && u < E)
                y = exp2_poly(fmaf(s[u], c, -m));
            else
                y = exp2_hw(fmaf(s[u], c, -m));
            acc[u] += y;
        }
        m += dm;
    }
    float tot = 0.0f;
    #pragma unroll
    for (int u = 0; u < FLUSSI; u++) tot += acc[u];
    out[gid] = tot;
}

// ---------------------------------------------------------------------------
// Accuratezza: calcola 2^x con un metodo su un vettore di x
// ---------------------------------------------------------------------------
template <int METODO>
__global__ void valuta_exp2(const float *x, float *y, int n) {
    const uint32_t lane = lane_id();
    const int32_t *tab = nullptr;
    if constexpr (METODO == TAB) {
        __shared__ int32_t s_tab[(1 << K) * 32];
        carica_tabella(s_tab);
        tab = s_tab;
    }
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i];
    if constexpr (METODO == TAB)       y[i] = exp2_tab(fmaf(v, 32768.0f, MAGIC), tab, lane);
    else if constexpr (METODO == POLY) y[i] = exp2_poly(v);
    else                               y[i] = exp2_hw(v);
}

// ---------------------------------------------------------------------------
// Utilita' host
// ---------------------------------------------------------------------------
static int cmp_float(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

// Warm-up, poi REPS esecuzioni misurate con gli eventi CUDA: restituisce la mediana in secondi
template <typename Lancio>
static double tempo_gpu(Lancio lancia) {
    cudaEvent_t ev[REPS + 1];
    for (int r = 0; r <= REPS; r++) CUDA_CHECK(cudaEventCreate(&ev[r]));
    lancia();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev[0]));
    for (int r = 0; r < REPS; r++) {
        lancia();
        CUDA_CHECK(cudaEventRecord(ev[r + 1]));
    }
    CUDA_CHECK(cudaEventSynchronize(ev[REPS]));
    CUDA_CHECK(cudaGetLastError());
    float ms[REPS];
    for (int r = 0; r < REPS; r++) CUDA_CHECK(cudaEventElapsedTime(&ms[r], ev[r], ev[r + 1]));
    for (int r = 0; r <= REPS; r++) CUDA_CHECK(cudaEventDestroy(ev[r]));
    qsort(ms, REPS, sizeof(float), cmp_float);
    return ms[REPS / 2] / 1000.0;
}

template <int METODO>
static void accuratezza(const char *nome, const float *dx, float *dy, const float *hx, float *hy, int n) {
    valuta_exp2<METODO><<<(n + 255) / 256, 256>>>(dx, dy, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hy, dy, n * sizeof(float), cudaMemcpyDeviceToHost));
    double emax = 0, esomma = 0, esegno = 0;
    for (int i = 0; i < n; i++) {
        double e = hy[i] / exp2((double)hx[i]) - 1.0;
        esegno += e;
        esomma += fabs(e);
        if (fabs(e) > emax) emax = fabs(e);
    }
    printf("%-26s errore relativo max %.2e   medio %.2e   medio con segno %+.1e\n",
           nome, emax, esomma / n, esegno / n);
}

struct Contesto {
    const float *dS;
    float *dOut, *hOut;
    int blocchi, sm;
    double clock_hz, n_exp;
    double t_mufu, somma_mufu;
};

template <int METODO, int E>
static void misura(Contesto &cx, const char *nome) {
    const float c = 1.44269504f;               // log2(e): la softmax usa e^x = 2^(x * log2 e)
    const float dm = 8.0f / PASSI;             // potenza di 2: m e m*2^15 restano esatti
    double t = tempo_gpu([&] {
        softmax_exp2<METODO, E><<<cx.blocchi, BLOCCO>>>(cx.dS, cx.dOut, c, dm);
    });
    int nt = cx.blocchi * BLOCCO;
    CUDA_CHECK(cudaMemcpy(cx.hOut, cx.dOut, nt * sizeof(float), cudaMemcpyDeviceToHost));
    double somma = 0;
    for (int i = 0; i < nt; i++) somma += cx.hOut[i];
    if (METODO == MUFU) { cx.t_mufu = t; cx.somma_mufu = somma; }

    double gexp = cx.n_exp / t / 1e9;
    double per_ciclo = cx.clock_hz > 0 ? cx.n_exp / t / cx.clock_hz / cx.sm : 0;
    printf("%-12s %d/8   %9.3f   %8.1f   %10.1f   %6.2fx   %+.1e\n", nome, METODO == MUFU ? 0 : E,
           t * 1e3, gexp, per_ciclo, cx.t_mufu / t, somma / cx.somma_mufu - 1.0);
}

int main(void) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int clock_khz = 0;
    if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0) != cudaSuccess) clock_khz = 0;
    printf("GPU: %s (%d SM, cc %d.%d, clock nominale %.0f MHz)\n\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor, clock_khz / 1000.0);

    // Tabella di correzione: centro della cella, come nei vostri file, meno 2^23 (vedi exp2_tab)
    int32_t tab[1 << K];
    for (int i = 0; i < (1 << K); i++) {
        double x = (i + 0.5) / (1 << K);
        tab[i] = (int32_t)lround((exp2(x) - 1 - x) * (1 << 23)) - (1 << 23);
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_tab, tab, sizeof tab));

    // ---------------- Accuratezza su x in [-20, 0], il campo tipico della softmax ----------------
    const int n = 1 << 20;
    float *hx = (float *)malloc(n * sizeof(float)), *hy = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) hx[i] = -20.0f + 20.0f * (float)i / (n - 1);
    float *dx, *dy;
    CUDA_CHECK(cudaMalloc(&dx, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx, n * sizeof(float), cudaMemcpyHostToDevice));
    printf("=== ACCURATEZZA di 2^x per x in [-20, 0] ===\n");
    accuratezza<MUFU>("MUFU (ex2.approx)", dx, dy, hx, hy, n);
    accuratezza<TAB>("tabella (trucco dei bit)", dx, dy, hx, hy, n);
    accuratezza<POLY>("polinomio (come FA4)", dx, dy, hx, hy, n);

    // ---------------- Throughput ----------------
    Contesto cx = {};
    cx.sm = prop.multiProcessorCount;
    cx.blocchi = cx.sm * BLOCCHI_PER_SM;
    cx.clock_hz = clock_khz * 1000.0;
    int nt = cx.blocchi * BLOCCO;
    cx.n_exp = (double)nt * PASSI * FLUSSI;

    float *hS = (float *)malloc((size_t)nt * FLUSSI * sizeof(float));
    srand(42);
    for (size_t i = 0; i < (size_t)nt * FLUSSI; i++)
        hS[i] = -5.5f * (float)rand() / RAND_MAX;     // punteggi gia' meno il massimo: x in [-16, 0]
    float *dS;
    CUDA_CHECK(cudaMalloc(&dS, (size_t)nt * FLUSSI * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cx.dOut, nt * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dS, hS, (size_t)nt * FLUSSI * sizeof(float), cudaMemcpyHostToDevice));
    cx.dS = dS;
    cx.hOut = (float *)malloc(nt * sizeof(float));

    printf("\n=== THROUGHPUT: softmax sintetica, %.2f miliardi di exp2 per esecuzione ===\n", cx.n_exp / 1e9);
    printf("metodo       sw    tempo [ms]   Gexp2/s   exp2/ciclo/SM   vs MUFU   diff. somma\n");
    misura<MUFU, 0>(cx, "MUFU");
    misura<TAB, 1>(cx, "tabella");
    misura<TAB, 2>(cx, "tabella");
    misura<TAB, 3>(cx, "tabella");
    misura<TAB, 4>(cx, "tabella");
    misura<TAB, 5>(cx, "tabella");
    misura<TAB, 6>(cx, "tabella");
    misura<TAB, 7>(cx, "tabella");
    misura<TAB, 8>(cx, "tabella");
    misura<POLY, 1>(cx, "polinomio");
    misura<POLY, 2>(cx, "polinomio");
    misura<POLY, 3>(cx, "polinomio");
    misura<POLY, 4>(cx, "polinomio");
    misura<POLY, 5>(cx, "polinomio");
    misura<POLY, 6>(cx, "polinomio");
    misura<POLY, 7>(cx, "polinomio");
    misura<POLY, 8>(cx, "polinomio");
    printf("\nsw = esponenziali su 8 calcolati in software (gli altri sulla MUFU).\n");
    printf("exp2/ciclo/SM e' calcolato al clock nominale: se sotto carico la GPU gira piu' bassa,\n");
    printf("il valore vero e' piu' alto. Il limite teorico della sola MUFU e' 16 per ciclo per SM.\n");

    cudaFree(dx); cudaFree(dy); cudaFree(dS); cudaFree(cx.dOut);
    free(hx); free(hy); free(hS); free(cx.hOut);
    return 0;
}
