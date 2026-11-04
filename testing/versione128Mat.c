#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#define BIAS_128 ((__int128)16383 << 112)

__int128 static inline fast_log2(__float128 x) {
    __int128 i;
    memcpy(&i, &x, sizeof i);
    return i - BIAS_128;
}

__float128 static inline fast_exp2(__int128 x) {
    __int128 i = x + BIAS_128;
    __float128 y;
    memcpy(&y, &i, sizeof y);
    return y;
}

__float128 static casuale(double minimo, double massimo) {
    const __float128 scala = 1.0Q / ((__float128)RAND_MAX + 1);
    __float128 u = (((__float128)rand() * scala + (__float128)rand()) * scala + (__float128)rand()) * scala;
    return (__float128)minimo + ((__float128)massimo - (__float128)minimo) * u;
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

    if (minimo <= 0 || massimo <= minimo) {
        printf("Servono due numeri positivi, con il massimo maggiore del minimo.\n");
        return 1;
    }
    __float128 *A = (__float128 *)malloc((size_t)n * n * sizeof(__float128));
    __float128 *B = (__float128 *)malloc((size_t)n * n * sizeof(__float128));
    __float128 *C_normal = (__float128 *)malloc((size_t)n * n * sizeof(__float128));
    __float128 *C_log = (__float128 *)malloc((size_t)n * n * sizeof(__float128));


    memset(C_normal, 0, (size_t)n * n * sizeof(__float128));
    memset(C_log, 0, (size_t)n * n * sizeof(__float128));

    srand(42);
    for (int i = 0; i < n * n; i++) {
        A[i] = casuale(minimo, massimo);
        B[i] = casuale(minimo, massimo);
    }
    clock_t start_normal = clock();
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            __float128 a = A[i * n + k];
            for (int j = 0; j < n; j++) {
                C_normal[i * n + j] += a * B[k * n + j];
            }
        }
    }
    clock_t end_normal = clock();
    clock_t start_log = clock();
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            __int128 log_a = fast_log2(A[i * n + k]);
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


    double errore_max = 0, errore_somma = 0;
    int sottostimati = 0;
    for (int i = 0; i < n * n; i++) {
        double e = (double)((C_log[i] - C_normal[i]) / C_normal[i]);
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
