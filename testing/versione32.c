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
    printf("Inserisci un numero 1: ");
    scanf("%f", &numero1);
    printf("Inserisci un numero 2: ");
    scanf("%f", &numero2);
    int iterazioni = 2000000;
    float *input1 = (float *)malloc(iterazioni * sizeof(float));
    float *input2 = (float *)malloc(iterazioni * sizeof(float));
    float *output_normal = (float *)malloc(iterazioni * sizeof(float));
    float *output_log = (float *)malloc(iterazioni * sizeof(float));


    memset(output_normal, 0, iterazioni * sizeof(float));
    memset(output_log, 0, iterazioni * sizeof(float));
    for (int i = 0; i < iterazioni; i++) {
        input1[i] = numero1 + ((float)i * 0.000001f);
        input2[i] = numero2;
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
    double tempo_normal = (((double)(end_normal - start_normal)) / CLOCKS_PER_SEC);
    double tempo_log = (((double)(end_log - start_log)) / CLOCKS_PER_SEC);

    printf("\nTempo impiegato standard:    %f secondi\n", tempo_normal);
    printf("Tempo impiegato logaritmico: %f secondi\n", tempo_log);
    printf("Rapporto standard/logaritmico: %.1fx\n", tempo_normal / tempo_log);
    free(input1);
    free(input2);
    free(output_normal);
    free(output_log);
    return 0;
}
