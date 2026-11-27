# Il trucco dei bit IEEE 754: calcolo rapido di $\log_2\vert{}d\vert{}$ e $\vert{}d\vert{}^p$

Questo documento spiega come sia possibile calcolare il logaritmo in base 2 e l'elevamento a potenza di un numero in virgola mobile (`float` a 32 bit) utilizzando esclusivamente operazioni aritmetiche su numeri interi e piccole tabelle di lookup, evitando le costose istruzioni trascendenti (`log2f`, `exp2f`, `powf`).

## 1. Come è fatto un `float` in memoria (IEEE 754)

Nello standard **IEEE 754 a singola precisione (32 bit)**, un numero `float` è memorizzato su 4 byte suddivisi in tre campi:

| Bit 31 (1 bit) | Bit 30 – 23 (8 bit) | Bit 22 – 0 (23 bit) | 
| ----- | ----- | ----- | 
| **Segno (**$S$**)** | **Esponente con bias (**$E$**)** | **Mantissa frazionaria (**$m$**)** | 

* **Segno (**$S$**):** vale `0` per i numeri positivi, `1` per i negativi.

* **Esponente (**$E$**):** è un intero senza segno compreso tra $0$ e $255$, spostato (*biased*) di $127$. L'esponente reale del numero è $E - 127$.

* **Mantissa (**$m$**):** sono i 23 bit dopo la virgola di un numero binario normalizzato del tipo $1.\text{xxxxxxx}_2$. Il suo valore reale frazionario è:
  

  $$
  M = \frac{m}{2^{23}} \in [0, 1)
  $$

Per qualsiasi numero normalizzato ($1 \le E \le 254$), il valore reale rappresentato è:

$$
\vert{}d\vert{} = (1 + M) \cdot 2^{E - 127}
$$

## 2. Leggere i bit di un `float` come un intero

Cosa succede se prendiamo gli stessi 32 bit in memoria di un numero positivo $\vert{}d\vert{}$ (con bit di segno $S = 0$) e li interpretiamo come un normale numero intero a 32 bit $I_d$?

* Il campo $E$ inizia al bit 23: in binario posizionale, il bit 23 ha peso $2^{23}$. Quindi il campo $E$ contribuisce al valore intero per $E \cdot 2^{23}$.

* Il campo $m$ occupa i 23 bit più bassi (peso $2^0 = 1$), e sappiamo che $m = M \cdot 2^{23}$.

Sommando i due contributi otteniamo l'identità fondamentale:

$$
I_d = E \cdot 2^{23} + m = E \cdot 2^{23} + M \cdot 2^{23} = (E + M) \cdot 2^{23}
$$

In altre parole, **l'intero** $I_d$ **è già uguale alla somma** $(E + M)$ **scalata di un fattore** $2^{23}$ (cioè in aritmetica in virgola fissa con 23 bit frazionari).

## 3. Dal valore intero $I_d$ al $\log_2\vert{}d\vert{}$

Applichiamo ora il logaritmo in base 2 alla formula del valore reale $\vert{}d\vert{}$:

$$
\log_2\vert{}d\vert{} = \log_2\left((1 + M) \cdot 2^{E - 127}\right) = (E - 127) + \log_2(1 + M)
$$

Osserviamo la funzione $\log_2(1 + M)$ nell'intervallo $M \in [0, 1)$:

* Per $M = 0$: $\log_2(1) = 0$

* Per $M \to 1$: $\log_2(2) = 1$

La curva $\log_2(1 + M)$ è molto vicina alla retta $y = M$: scarta al massimo di circa $0.086$ (attorno a $M \approx 0.44$). Possiamo quindi riscriverla separando la parte lineare dall'errore:

$$
\log_2(1 + M) = M + \underbrace{\left(\log_2(1 + M) - M\right)}_{\mathcal{E}_{\log}(M)}
$$

Sostituendo questa espressione nella formula del logaritmo:

$$
\log_2\vert{}d\vert{} = (E - 127) + M + \mathcal{E}_{\log}(M) = (E + M) - 127 + \mathcal{E}_{\log}(M)
$$

Moltiplichiamo entrambi i membri per $2^{23}$ per passare alla rappresentazione intera in virgola fissa:

$$
\log_2\vert{}d\vert{} \cdot 2^{23} = \underbrace{(E + M) \cdot 2^{23}}_{I_d} - \underbrace{127 \cdot 2^{23}}_{\text{BIAS\_32}} + \underbrace{\mathcal{E}_{\log}(M) \cdot 2^{23}}_{\text{correzione}} = I_d - \text{BIAS\_32} + \text{correzione}
$$

Questa formula dimostra che, a meno di una piccola correzione tabellata, **i bit interi di un `float` sono già il suo logaritmo in base 2**.

## 4. Calcolo delle due tabelle di correzione (`h_log` e `h_exp`)

Poiché $M$ ha $2^{23} = 8\,388\,608$ valori possibili, non possiamo memorizzare l'errore per ogni singolo valore di $M$. Dividiamo invece l'intervallo $[0, 1)$ in $N_T = 2^K$ celle uniformi (nel codice $K = 7 \implies N_T = 128$ voci), indicizzate dai $K$ **bit più significativi della mantissa**.

Per la cella $i$-esima ($i \in \{0, 1, \dots, 127\}$), il punto centrale dell'intervallo è:

$$
x_i = \frac{i + 0.5}{N_T}
$$

### A. Tabella del logaritmo (`tab_log`)

Per risparmiare un'operazione a runtime, nella tabella del logaritmo si ingloba direttamente la sottrazione del bias ($\text{BIAS\_32} = 127 \cdot 2^{23} = 1\,065\,353\,216$):

$$
\text{tab\_log}[i] = \text{round}\Big(\big(\log_2(1 + x_i) - x_i\big) \cdot 2^{23}\Big) - 127 \cdot 2^{23}
$$

A runtime, il calcolo di $L \approx \log_2\vert{}d\vert{} \cdot 2^{23}$ si riduce a una singola somma intera:

$$
L = I_d + \text{tab\_log}[i]
$$

### B. Tabella dell'esponenziale inverso (`tab_exp`)

Una volta moltiplicato $L$ per l'esponente $p$, otteniamo $L_p \approx (p \cdot \log_2\vert{}d\vert{}) \cdot 2^{23}$. Dobbiamo ora calcolare $2^{L_p / 2^{23}}$ per tornare a un `float`.

Scomponiamo $y = L_p / 2^{23}$ nella sua parte intera $E' = \lfloor y \rfloor$ e nella sua parte frazionaria $f \in [0, 1)$:

$$
2^y = 2^{E' + f} = 2^f \cdot 2^{E'}
$$

Confrontando con la formula IEEE 754 $(1 + M_{\text{out}}) \cdot 2^{E_{\text{out}} - 127}$, ricaviamo che:

* Il nuovo esponente con bias è $E_{\text{out}} = E' + 127$.

* La nuova mantissa è $M_{\text{out}} = 2^f - 1 = f + (2^f - 1 - f)$.

Moltiplicando per $2^{23}$ e sommando esponente e mantissa per ricostruire i bit interi del `float` risultato:

$$
I_{\text{out}} = (E_{\text{out}} + M_{\text{out}}) \cdot 2^{23} = \underbrace{(E' + f) \cdot 2^{23}}_{L_p} + \underbrace{127 \cdot 2^{23}}_{\text{BIAS\_32}} + \underbrace{(2^f - 1 - f) \cdot 2^{23}}_{\text{correzione exp}}
$$

Pertanto, la seconda tabella (indicizzata dai $K$ bit più alti della parte frazionaria di $L_p$) è definita come:

$$
\text{tab\_exp}[i] = \text{round}\Big(\big(2^{x_i} - 1 - x_i\big) \cdot 2^{23}\Big) + 127 \cdot 2^{23}
$$

A runtime, ricostruire i bit del `float` risultato richiede di nuovo una sola somma:

$$
I_{\text{out}} = L_p + \text{tab\_exp}[j]
$$

## 5. Esempi numerici passo dopo passo

Consideriamo i parametri del codice: $K = 7$, $N_T = 128$, $2^{23} = 8\,388\,608$, $\text{BIAS\_32} = 1\,065\,353\,216$.

### Esempio 1: Calcolo di $\log_2(6.0)$

1. **Rappresentazione IEEE 754 di** $d = 6.0$**:**

   * $6.0 = 1.5 \times 2^2 = (1 + 0.5) \cdot 2^{129 - 127}$

   * Segno $S = 0$, Esponente $E = 129$, Mantissa $M = 0.5$ ($m = 0.5 \cdot 2^{23} = 4\,194\,304$).

   * Letto come intero:
     

     $$
     I_d = 129 \cdot 2^{23} + 4\,194\,304 = 1\,082\,130\,432 + 4\,194\,304 = 1\,086\,324\,736
     $$

     
     *(in esadecimale: `0x40C00000`)*.

2. **Indice nella tabella `tab_log`:**

   * I $K = 7$ bit più alti della mantissa ($M = 0.5 = 0.1000000_2$) danno l'indice:
     

     $$
     i = 64
     $$

   * Il centro della cella $64$ è $x_{64} = \frac{64.5}{128} \approx 0.503906$.

   * L'errore precalcolato in quella cella è:
     

     $$
     \text{corr} = \text{round}\big((\log_2(1.503906) - 0.503906) \cdot 2^{23}\big) = \text{round}(0.084962 \cdot 8\,388\,608) = 712\,717
     $$

   * Quindi $\text{tab\_log}[64] = 712\,717 - 1\,065\,353\,216 = -1\,064\,640\,499$.

3. **Somma finale e verifica:**
   

   $$
   L = I_d + \text{tab\_log}[64] = 1\,086\,324\,736 - 1\,064\,640\,499 = 21\,684\,237
   $$

   
   Dividendo per $2^{23}$ per leggere il risultato reale:
   

   $$
   \frac{21\,684\,237}{8\,388\,608} \approx \mathbf{2.584962}
   $$

   
   *(Il valore esatto di* $\log_2(6)$ *è* $\mathbf{2.5849625}$*: l'errore è inferiore a* $10^{-6}$*!).*

### Esempio 2: Calcolo completo di $\vert{}d\vert{}^p$ con $d = 6.0$ e $p = 0.5$ (radice quadrata $\sqrt{6}$)

Partiamo dal valore $L = 21\,684\,237$ appena trovato per $\log_2(6.0)$.

1. **Moltiplicazione in virgola fissa per** $p = 0.5$**:**

   * Nel codice $P = \text{round}(0.5 \cdot 2^{28}) = 134\,217\,728$.

   * Calcoliamo $q = \lfloor (L \cdot P) / 2^{32} \rfloor$:
     

     $$
     q = \left\lfloor \frac{21\,684\,237 \cdot 134\,217\,728}{4\,294\,967\,296} \right\rfloor = \lfloor 677\,632.4 \rfloor = 677\,632
     $$

   * Riportiamo alla scala $2^{23}$ moltiplicando per $16$:
     

     $$
     L_p = 677\,632 \cdot 16 = 10\,842\,112
     $$

     
     *(Nota:* $10\,842\,112 / 2^{23} \approx 1.29248$*, che è esattamente* $\frac{1}{2}\log_2(6)$*).*

2. **Indice nella tabella `tab_exp`:**

   * Estraiamo i $K = 7$ bit più alti della parte frazionaria di $L_p$ (i bit dal 22 al 16 di $L_p$):
     

     $$
     j = (10\,842\,112 \gg 16) \ \& \ 127 = 165 \ \& \ 127 = 37
     $$

   * Il centro della cella $37$ è $x_{37} = \frac{37.5}{128} \approx 0.292969$.

   * La correzione dell'esponenziale in quella cella è:
     

     $$
     \text{corr}_{\exp} = \text{round}\big((2^{0.292969} - 1 - 0.292969) \cdot 2^{23}\big) = \text{round}(-0.067799 \cdot 8\,388\,608) = -568\,743
     $$

   * Quindi $\text{tab\_exp}[37] = -568\,743 + 1\,065\,353\,216 = 1\,064\,784\,473$.

3. **Ricostruzione del `float` risultato:**
   

   $$
   I_{\text{out}} = L_p + \text{tab\_exp}[37] = 10\,842\,112 + 1\,064\,784\,473 = 1\,075\,626\,585
   $$

   
   Interpretiando $1\,075\,626\,585$ (`0x401CFFFF` circa) come `float` IEEE 754:

   * Esponente: $\lfloor 1\,075\,626\,585 / 2^{23} \rfloor = 128 \implies 2^{128 - 127} = 2^1$.

   * Mantissa: $(1\,075\,626\,585 \bmod 2^{23}) / 2^{23} = 1\,884\,761 / 8\,388\,608 \approx 0.224681$.

   * Valore finale:
     

     $$
     y = (1 + 0.224681) \cdot 2^1 \approx \mathbf{2.44936}
     $$

   *(Il valore esatto di* $\sqrt{6}$ *è* $\mathbf{2.44949}$*: abbiamo ottenuto il risultato corretto fino alla quarta cifra decimale usando solo 3 somme, 1 moltiplicazione intera e 2 letture in tabella).*