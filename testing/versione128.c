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
    printf("Inserisci un numero 1: ");
    scanf("%lf", &numero1);
    printf("Inserisci un numero 2: ");
    scanf("%lf", &numero2);
    int iterazioni = 2000000;
    __float128 *input1 = (__float128 *)malloc(iterazioni * sizeof(__float128));
    __float128 *input2 = (__float128 *)malloc(iterazioni * sizeof(__float128));
    __float128 *output_normal = (__float128 *)malloc(iterazioni * sizeof(__float128));
    __float128 *output_log = (__float128 *)malloc(iterazioni * sizeof(__float128));


    memset(output_normal, 0, iterazioni * sizeof(__float128));
    memset(output_log, 0, iterazioni * sizeof(__float128));
    for (int i = 0; i < iterazioni; i++) {
        input1[i] = (__float128)numero1 + ((__float128)i * 0.000001Q);
        input2[i] = (__float128)numero2;
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
