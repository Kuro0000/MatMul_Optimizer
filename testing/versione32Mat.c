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

int main(void) {
    float numero1, numero2;
    int n;
    printf("Inserisci un numero 1: ");
    scanf("%f", &numero1);
    printf("Inserisci un numero 2: ");
    scanf("%f", &numero2);
    printf("Inserisci la dimensione N delle matrici: ");
    scanf("%d", &n);
    float *A = (float *)malloc((size_t)n * n * sizeof(float));
    float *B = (float *)malloc((size_t)n * n * sizeof(float));
    float *C_normal = (float *)malloc((size_t)n * n * sizeof(float));
    float *C_log = (float *)malloc((size_t)n * n * sizeof(float));


    memset(C_normal, 0, (size_t)n * n * sizeof(float));
    memset(C_log, 0, (size_t)n * n * sizeof(float));
    for (int i = 0; i < n * n; i++) {
        A[i] = numero1 + ((float)(i % 1000) * 0.001f);
        B[i] = numero2 + ((float)(i % 777) * 0.001f);
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
    free(A);
    free(B);
    free(C_normal);
    free(C_log);
    return 0;
}
