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

int main(void) {
    double numero1, numero2;
    int n;
    printf("Inserisci un numero 1: ");
    scanf("%lf", &numero1);
    printf("Inserisci un numero 2: ");
    scanf("%lf", &numero2);
    printf("Inserisci la dimensione N delle matrici: ");
    scanf("%d", &n);
    __float128 *A = (__float128 *)malloc((size_t)n * n * sizeof(__float128));
    __float128 *B = (__float128 *)malloc((size_t)n * n * sizeof(__float128));
    __float128 *C_normal = (__float128 *)malloc((size_t)n * n * sizeof(__float128));
    __float128 *C_log = (__float128 *)malloc((size_t)n * n * sizeof(__float128));


    memset(C_normal, 0, (size_t)n * n * sizeof(__float128));
    memset(C_log, 0, (size_t)n * n * sizeof(__float128));
    for (int i = 0; i < n * n; i++) {
        A[i] = (__float128)numero1 + ((__float128)(i % 1000) * 0.001Q);
        B[i] = (__float128)numero2 + ((__float128)(i % 777) * 0.001Q);
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
    free(A);
    free(B);
    free(C_normal);
    free(C_log);
    return 0;
}
