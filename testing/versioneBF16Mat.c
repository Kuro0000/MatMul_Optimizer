#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define BIAS_BF16 ((int32_t)127 << 7)
#define K 7

static int16_t Elog[1 << K], Eexp[1 << K];

void init_tables(void) {
    double offset = (K < 7) ? 0.5 : 0.0;
    for (int i = 0; i < (1 << K); i++) {
        double x = (i + offset) / (1 << K);
        Elog[i] = (int16_t)lround((log2(1.0 + x) - x) * (1 << 7));
        Eexp[i] = (int16_t)lround((exp2(x) - 1.0 - x) * (1 << 7));
    }
}

static inline int32_t fast_log2(__bf16 x) {
    uint16_t i;
    memcpy(&i, &x, sizeof i);
    i &= 0x7FFF;
    if (i < 0x0080)
        return -100000;
    uint16_t idx = (i >> (7 - K)) & ((1 << K) - 1);
    return ((int32_t)i - BIAS_BF16) + Elog[idx];
}

static inline __bf16 fast_exp2(int32_t L) {
    if (L < -16128) return (__bf16)0.0f;

    uint16_t idx = ((uint16_t)L >> (7 - K)) & ((1 << K) - 1);
    int32_t i = L + BIAS_BF16 + Eexp[idx];

    if (i < 0x0080)
        return (__bf16)0.0f;
    if (i >= 0x7F80) {
        uint16_t inf_bits = 0x7F80;
        __bf16 y;
        memcpy(&y, &inf_bits, sizeof y);
        return y;
    }

    uint16_t bits = (uint16_t)i;
    __bf16 y;
    memcpy(&y, &bits, sizeof y);
    return y;
}

static float casuale(double minimo, double massimo) {
    return (float)(minimo + (massimo - minimo) * ((double)rand() / RAND_MAX));
}

static void matmul_normal(const __bf16 *A, const __bf16 *B, __bf16 *C,
                          float *acc, int N) {
    size_t total = (size_t)N * N;
    memset(acc, 0, total * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a = (float)A[(size_t)i * N + k];
            const __bf16 *rowB = &B[(size_t)k * N];
            float *rowAcc = &acc[(size_t)i * N];
            for (int j = 0; j < N; j++) {
                rowAcc[j] += a * (float)rowB[j];
            }
        }
    }

    for (size_t idx = 0; idx < total; idx++) {
        C[idx] = (__bf16)acc[idx];
    }
}

static void matmul_log(const __bf16 *A, const __bf16 *B, __bf16 *C,
                       int32_t *logA, int32_t *logB, float *acc, int N) {
    size_t total = (size_t)N * N;

    // 1. Pre-calcolo dei logaritmi: costo O(N^2) ammortizzato su O(N^3) MAC
    for (size_t idx = 0; idx < total; idx++) {
        logA[idx] = fast_log2(A[idx]);
        logB[idx] = fast_log2(B[idx]);
    }

    // 2. Prodotto-accumulo: somma nel dominio logaritmico, accumulo lineare in FP32
    memset(acc, 0, total * sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            int32_t la = logA[(size_t)i * N + k];
            const int32_t *rowB = &logB[(size_t)k * N];
            float *rowAcc = &acc[(size_t)i * N];
            for (int j = 0; j < N; j++) {
                rowAcc[j] += (float)fast_exp2(la + rowB[j]);
            }
        }
    }

    // 3. Quantizzazione finale in BF16
    for (size_t idx = 0; idx < total; idx++) {
        C[idx] = (__bf16)acc[idx];
    }
}

int main(void) {
    int N;
    double minimo, massimo;
    init_tables();

    printf("Inserisci la dimensione N delle matrici NxN (es. 256 o 512): ");
    if (scanf("%d", &N) != 1 || N <= 0) return 1;
    printf("Inserisci il valore minimo: ");
    if (scanf("%lf", &minimo) != 1) return 1;
    printf("Inserisci il valore massimo: ");
    if (scanf("%lf", &massimo) != 1) return 1;

    if (minimo <= 0 || massimo <= minimo) {
        printf("Servono due numeri positivi, con il massimo maggiore del minimo.\n");
        return 1;
    }

    size_t total = (size_t)N * N;
    __bf16 *A = malloc(total * sizeof(__bf16));
    __bf16 *B = malloc(total * sizeof(__bf16));
    __bf16 *C_normal = malloc(total * sizeof(__bf16));
    __bf16 *C_log = malloc(total * sizeof(__bf16));
    float *acc = malloc(total * sizeof(float));
    int32_t *logA = malloc(total * sizeof(int32_t));
    int32_t *logB = malloc(total * sizeof(int32_t));

    if (!A || !B || !C_normal || !C_log || !acc || !logA || !logB) {
        printf("Errore di allocazione memoria.\n");
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < total; i++) {
        A[i] = (__bf16)casuale(minimo, massimo);
        B[i] = (__bf16)casuale(minimo, massimo);
    }

    matmul_normal(A, B, C_normal, acc, N);
    matmul_log(A, B, C_log, logA, logB, acc, N);

    clock_t start_normal = clock();
    matmul_normal(A, B, C_normal, acc, N);
    clock_t end_normal = clock();

    clock_t start_log = clock();
    matmul_log(A, B, C_log, logA, logB, acc, N);
    clock_t end_log = clock();

    double tempo_normal = (double)(end_normal - start_normal) / CLOCKS_PER_SEC;
    double tempo_log = (double)(end_log - start_log) / CLOCKS_PER_SEC;
    double ops = 2.0 * (double)N * (double)N * (double)N;

    printf("\n--- PRESTAZIONI GEMM %dx%d (%zu MAC) ---\n", N, N, (size_t)N * N * N);
    printf("Tempo impiegato standard:      %f s (%.2f GFLOPS)\n",
           tempo_normal, (ops / tempo_normal) / 1e9);
    printf("Tempo impiegato logaritmico:   %f s (%.2f GFLOPS)\n",
           tempo_log, (ops / tempo_log) / 1e9);
    printf("Rapporto standard/logaritmico: %.2fx\n", tempo_normal / tempo_log);

    double errore_max = 0.0, errore_somma = 0.0;
    size_t sottostimati = 0, errori_su_zero = 0, validi = 0;
    for (size_t i = 0; i < total; i++) {
        double ref = (double)C_normal[i];
        double approx = (double)C_log[i];
        if (ref == 0.0) {
            if (approx != 0.0) errori_su_zero++;
            continue;
        }
        validi++;
        double e = (approx - ref) / ref;
        errore_somma += e;
        if (e < 0.0) sottostimati++;
        double assoluto = fabs(e);
        if (assoluto > errore_max) errore_max = assoluto;
    }

    printf("\n--- ERRORE RELATIVO SU MATRICE C (%zu elementi) ---\n", total);
    printf("Errore massimo:          %.4f%%\n", 100.0 * errore_max);
    printf("Errore medio:            %+.4f%%\n", validi > 0 ? 100.0 * errore_somma / validi : 0.0);
    printf("Risultati sottostimati:  %zu su %zu\n", sottostimati, validi);
    if (errori_su_zero > 0) {
        printf("Errori su zero:          %zu\n", errori_su_zero);
    }

    free(A);
    free(B);
    free(C_normal);
    free(C_log);
    free(acc);
    free(logA);
    free(logB);
    return 0;
}
