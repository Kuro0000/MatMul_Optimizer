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

int main(void) {
    double minimo, massimo;
    init_tables();

    printf("Inserisci il valore minimo: ");
    if (scanf("%lf", &minimo) != 1) return 1;
    printf("Inserisci il valore massimo: ");
    if (scanf("%lf", &massimo) != 1) return 1;

    if (minimo <= 0 || massimo <= minimo) {
        printf("Servono due numeri positivi, con il massimo maggiore del minimo.\n");
        return 1;
    }

    int iterazioni = 2000000;
    __bf16 *input1_bf = malloc(iterazioni * sizeof(__bf16));
    __bf16 *input2_bf = malloc(iterazioni * sizeof(__bf16));
    __bf16 *output_normal = malloc(iterazioni * sizeof(__bf16));
    __bf16 *output_log = malloc(iterazioni * sizeof(__bf16));

    if (!input1_bf || !input2_bf || !output_normal || !output_log) return 1;

    memset(output_normal, 0, iterazioni * sizeof(__bf16));
    memset(output_log, 0, iterazioni * sizeof(__bf16));

    srand(42);
    for (int i = 0; i < iterazioni; i++) {
        input1_bf[i] = (__bf16)casuale(minimo, massimo);
        input2_bf[i] = (__bf16)casuale(minimo, massimo);
    }

    clock_t start_normal = clock();
    for (int i = 0; i < iterazioni; i++) {
        output_normal[i] = (__bf16)((float)input1_bf[i] * (float)input2_bf[i]);
    }
    clock_t end_normal = clock();

    clock_t start_log = clock();
    for (int i = 0; i < iterazioni; i++) {
        output_log[i] = fast_exp2(fast_log2(input1_bf[i]) + fast_log2(input2_bf[i]));
    }
    clock_t end_log = clock();

    double tempo_normal = (double)(end_normal - start_normal) / CLOCKS_PER_SEC;
    double tempo_log = (double)(end_log - start_log) / CLOCKS_PER_SEC;

    printf("\nTempo impiegato standard:    %f secondi\n", tempo_normal);
    printf("Tempo impiegato logaritmico: %f secondi\n", tempo_log);
    printf("Rapporto standard/logaritmico: %.2fx\n", tempo_normal / tempo_log);

    double errore_max = 0.0, errore_somma = 0.0;
    int sottostimati = 0, errori_su_zero = 0, validi = 0;
    for (int i = 0; i < iterazioni; i++) {
        double ref = (double)output_normal[i];
        double approx = (double)output_log[i];
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

    printf("\n--- ERRORE RELATIVO (BF16 vs BF16) ---\n");
    printf("Errore massimo:          %.4f%%\n", 100.0 * errore_max);
    printf("Errore medio:            %+.4f%%\n", validi > 0 ? 100.0 * errore_somma / validi : 0.0);
    printf("Risultati sottostimati:  %d su %d\n", sottostimati, validi);
    if (errori_su_zero > 0) {
        printf("Errori su zero:          %d\n", errori_su_zero);
    }

    free(input1_bf);
    free(input2_bf);
    free(output_normal);
    free(output_log);
    return 0;
}
