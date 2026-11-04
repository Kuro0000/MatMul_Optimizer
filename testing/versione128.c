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

// Numero casuale tra minimo e massimo: tre chiamate a rand() riempiono quasi tutti i 112 bit della mantissa
__float128 static casuale(double minimo, double massimo) {
    const __float128 scala = 1.0Q / ((__float128)RAND_MAX + 1);
    __float128 u = (((__float128)rand() * scala + (__float128)rand()) * scala + (__float128)rand()) * scala;
    return (__float128)minimo + ((__float128)massimo - (__float128)minimo) * u;
}

int main(void) {
    double minimo, massimo;
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
    __float128 *input1 = (__float128 *)malloc(iterazioni * sizeof(__float128));
    __float128 *input2 = (__float128 *)malloc(iterazioni * sizeof(__float128));
    __float128 *output_normal = (__float128 *)malloc(iterazioni * sizeof(__float128));
    __float128 *output_log = (__float128 *)malloc(iterazioni * sizeof(__float128));


    memset(output_normal, 0, iterazioni * sizeof(__float128));
    memset(output_log, 0, iterazioni * sizeof(__float128));
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
    double tempo_normal = (((double)(end_normal - start_normal)) / CLOCKS_PER_SEC);
    double tempo_log = (((double)(end_log - start_log)) / CLOCKS_PER_SEC);

    printf("\nTempo impiegato standard:    %f secondi\n", tempo_normal);
    printf("Tempo impiegato logaritmico: %f secondi\n", tempo_log);
    printf("Rapporto standard/logaritmico: %.1fx\n", tempo_normal / tempo_log);

    // --- CALCOLO DELL'ERRORE RELATIVO: (logaritmico - standard) / standard ---
    // Con __float128 non serve fabs: il valore assoluto si calcola a mano
    double errore_max = 0, errore_somma = 0;
    int sottostimati = 0;
    for (int i = 0; i < iterazioni; i++) {
        double e = (double)((output_log[i] - output_normal[i]) / output_normal[i]);
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
