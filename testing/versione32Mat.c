#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#define BIAS_32 ((int32_t)127 << 23)

int32_t static inline fast_log2(float x) {
    int32_t i;
    memcpy(&i, &x, sizeof i);
    return i - BIAS_32;
}

float static inline fast_exp2(int32_t x) {
    int32_t i = x + BIAS_32;
    float y;
    memcpy(&y, &i, sizeof y);
    return y;
}

// Numero casuale tra minimo e massimo (calcolato in double e poi convertito in float)
float static casuale(double minimo, double massimo) {
    return (float)(minimo + (massimo - minimo) * ((double)rand() / RAND_MAX));
}

int main(void) {
    double minimo, massimo;
    int n;
    printf("Inserisci il valore minimo: ");
    scanf("%lf", &minimo);
    printf("Inserisci il valore massimo: ");
    scanf("%lf", &massimo);
    printf("Inserisci la dimensione N delle matrici: ");
    scanf("%d", &n);
    // Il metodo vale solo per numeri positivi. Per coprire tutte le mantisse
    // il massimo dovrebbe essere almeno il doppio del minimo (es. 1 e 1000)
    if (minimo <= 0 || massimo <= minimo) {
        printf("Servono due numeri positivi, con il massimo maggiore del minimo.\n");
        return 1;
    }
    float *A = (float *)malloc((size_t)n * n * sizeof(float));
    float *B = (float *)malloc((size_t)n * n * sizeof(float));
    float *C_normal = (float *)malloc((size_t)n * n * sizeof(float));
    float *C_log = (float *)malloc((size_t)n * n * sizeof(float));


    memset(C_normal, 0, (size_t)n * n * sizeof(float));
    memset(C_log, 0, (size_t)n * n * sizeof(float));
    // Input casuali: con valori quasi uguali si misurerebbe l'errore di una sola coppia di mantisse
    srand(42);   // stesso seme: dati identici a ogni esecuzione
    for (int i = 0; i < n * n; i++) {
        A[i] = casuale(minimo, massimo);
        B[i] = casuale(minimo, massimo);
    }
    clock_t start_normal = clock();
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            float a = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C_normal[i * n + j] += a * B[k * n + j];
            }
        }
    }
    clock_t end_normal = clock();
    clock_t start_log = clock();
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            int32_t log_a = fast_log2(A[i * n + k]);
            for (int j = 0; j < n; j++) {
                C_log[i * n + j] += fast_exp2(log_a + fast_log2(B[k * n + j]));
            }
        }
    }
    clock_t end_log = clock();
    double tempo_normal = (((double)(end_normal - start_normal)) / CLOCKS_PER_SEC);
    double tempo_log = (((double)(end_log - start_log)) / CLOCKS_PER_SEC);

    printf("\nTempo impiegato standard:    %f secondi\n", tempo_normal);
    printf("Tempo impiegato logaritmico: %f secondi\n", tempo_log);
    printf("Rapporto standard/logaritmico: %.1fx\n", tempo_normal / tempo_log);

    // --- CALCOLO DELL'ERRORE RELATIVO SULLA MATRICE: (logaritmico - standard) / standard ---
    // Ogni elemento di C e' una somma di n prodotti, quindi l'errore va calcolato sul risultato finale
    double errore_max = 0, errore_somma = 0;
    int sottostimati = 0;
    for (int i = 0; i < n * n; i++) {
        double e = ((double)C_log[i] - C_normal[i]) / C_normal[i];
        errore_somma += e;
        if (e < 0) sottostimati++;
        double assoluto = e < 0 ? -e : e;
        if (assoluto > errore_max) errore_max = assoluto;
    }
    printf("\n--- ERRORE RELATIVO SULLA MATRICE ---\n");
    printf("Errore massimo:          %.4f%%\n", 100 * errore_max);
    printf("Errore medio:            %+.4f%%\n", 100 * errore_somma / ((double)n * n));
    printf("Elementi sottostimati:   %d su %d\n", sottostimati, n * n);

    free(A);
    free(B);
    free(C_normal);
    free(C_log);
    return 0;
}
