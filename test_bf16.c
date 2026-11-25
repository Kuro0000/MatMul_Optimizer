
#include <stdio.h>
#include <stdint.h>

int main() {
    // 1. Verifica la dimensione del tipo nativo
    printf("Dimensione di __bf16: %zu byte (Atteso: 2)\n", sizeof(__bf16));

    // 2. Inizializzazione di un valore bf16 nativo da un float
    float input_val = 3.14159f;
    __bf16 bf_val = (__bf16)input_val;

    // 3. Riconversione a float per la stampa a video
    float output_val = (float)bf_val;

    printf("Float originale: %f\n", input_val);
    printf("Valore convertito in __bf16 e riportato a float: %f\n", output_val);

    return 0;
}
