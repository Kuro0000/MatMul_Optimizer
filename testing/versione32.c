#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define BIAS_32 ((int32_t)127 << 23)
#define K 8

static int32_t Elog[1 << K], Eexp[1 << K];

void init_tables(void) {
    for (int i = 0; i < (1 << K); i++) {
        double x = (i + 0.5) / (1 << K);              // centro della cella
        Elog[i] = (int32_t)lround((log2(1 + x) - x) * (1 << 23));
        Eexp[i] = (int32_t)lround((exp2(x) - 1 - x) * (1 << 23));
    }
}

static inline int32_t fast_log2(float x) {
    int32_t i; memcpy(&i, &x, sizeof i);
    return (i - BIAS_32) + Elog[(i >> (23 - K)) & ((1 << K) - 1)];
}

static inline float fast_exp2(int32_t L) {
    int32_t i = L + BIAS_32 + Eexp[(L >> (23 - K)) & ((1 << K) - 1)];
    float y; memcpy(&y, &i, sizeof y);
    return y;
}

static inline float fast_mul(int32_t a, int32_t b) {
    int32_t ia; memcpy(&ia, &a, sizeof ia);
    int32_t ib; memcpy(&ib, &b, sizeof ib);
    int32_t sum = ia - BIAS_32 + Elog[(ia >> (23 - K)) & ((1 << K) - 1)] + ib - BIAS_32 + Elog[(ib >> (23 - K)) & ((1 << K) - 1)];

    int32_t isum = sum + BIAS_32 + Eexp[(sum >> (23 - K)) & ((1 << K) - 1)];
    float y; memcpy(&y, &isum, sizeof y);
    return y;
}

// Numero casuale tra minimo e massimo (calcolato in double e poi convertito in float)
float static casuale(double minimo, double massimo) {
    return (float)(minimo + (massimo - minimo) * ((double)rand() / RAND_MAX));
}

int main(void) {
    double minimo, massimo;
    init_tables();

    printf("Inserisci il valore minimo: ");
    scanf("%lf", &minimo);
    printf("Inserisci il valore massimo: ");
    scanf("%lf", &massimo);
    // Il metodo vale solo per numeri positivi. Per coprire tutte le mantisse
    // il massimo dovrebbe essere almeno il doppio del minimo (es. 1 e 1000)
    if (minimo <= 0 || massimo <= minimo) {
        printf("Servono due numeri positivi, con il massimo maggiore del minimo.\n");
        return 1;
    }
    int iterazioni = 2000000;
    float *input1 = (float *)malloc(iterazioni * sizeof(float));
    float *input2 = (float *)malloc(iterazioni * sizeof(float));
    float *output_normal = (float *)malloc(iterazioni * sizeof(float));
    float *output_log = (float *)malloc(iterazioni * sizeof(float));


    memset(output_normal, 0, iterazioni * sizeof(float));
    memset(output_log, 0, iterazioni * sizeof(float));
    // Input casuali: con valori quasi uguali si misurerebbe l'errore di una sola coppia di mantisse
    srand(42);   // stesso seme: dati identici a ogni esecuzione
    for (int i = 0; i < iterazioni; i++) {
        input1[i] = casuale(minimo, massimo);
        input2[i] = casuale(minimo, massimo);
    }
    clock_t start_normal = clock();
    for (int i = 0; i < iterazioni; i++) {
        output_normal[i] = input1[i] * input2[i];
    }
    clock_t end_normal = clock();
    clock_t start_log = clock();
    for (int i = 0; i < iterazioni; i++) {
        output_log[i] = fast_exp2(fast_log2(input1[i]) + fast_log2(input2[i]));
    }
    clock_t end_log = clock();
    clock_t start_fast_mul = clock();
    for (int i = 0; i < iterazioni; i++) {
        output_log[i] = fast_mul(input1[i], input2[i]);
    }
    clock_t end_fast_mul = clock();

    double tempo_normal = (((double)(end_normal - start_normal)) / CLOCKS_PER_SEC);
    double tempo_log = (((double)(end_log - start_log)) / CLOCKS_PER_SEC);
    double tempo_fast_mul = (((double)(end_fast_mul - start_fast_mul)) / CLOCKS_PER_SEC);

    printf("\nTempo impiegato standard:    %f secondi\n", tempo_normal);
    printf("Tempo impiegato logaritmico: %f secondi\n", tempo_log);
    printf("Tempo impiegato fast mul: %f secondi\n", tempo_fast_mul);
    printf("Rapporto standard/logaritmico: %.1fx\n", tempo_normal / tempo_log);

    // --- CALCOLO DELL'ERRORE RELATIVO: (logaritmico - standard) / standard ---
    double errore_max = 0, errore_somma = 0;
    int sottostimati = 0;
    for (int i = 0; i < iterazioni; i++) {
        double e = ((double)output_log[i] - output_normal[i]) / output_normal[i];
        errore_somma += e;
        if (e < 0) sottostimati++;
        double assoluto = e < 0 ? -e : e;
        if (assoluto > errore_max) errore_max = assoluto;
    }
    printf("\n--- ERRORE RELATIVO ---\n");
    printf("Errore massimo:          %.4f%%\n", 100 * errore_max);
    printf("Errore medio:            %+.4f%%\n", 100 * errore_somma / iterazioni);
    printf("Risultati sottostimati:  %d su %d\n", sottostimati, iterazioni);

    free(input1);
    free(input2);
    free(output_normal);
    free(output_log);
    return 0;
}
