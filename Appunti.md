# Appunti e note sui progressi

## Confronto con la moltiplicazione hardware sulla CPU

Il benchmark del prodotto tra matrici confronta un ciclo standard con lo stesso ciclo in cui la moltiplicazione è sostituita dal logBasedMultiply, compilando con `gcc -O3 -march=native`. La versione logaritmica pareggia la moltiplicazione standard.
Il perché è che la moltiplicazione float(un semplice a*b) ha già l'operazione più ottimizzata del processore, infatti viene eseguita da una singola istruzione e nel prodotto tra matrici l'operazione `C += a * b` diventa un'unica istruzione FMA in cui moltiplicazione e somma avvengono insieme, rendendo di fatto la moltiplicazione gratuita, infatti nel file versione32.c in cui non c'è il calcolo matriciale se la compiliamo e la eseguiamo vediamo che i risultati rimangono ottimi. logBasedMultiply invece richiede diverse istruzioni intere per la somma dei bit e la gestione del segno, seguite da una somma float separata. Il risultato è che sulla CPU(nella GPU è ancora da testare, ma prevedo pessime performance) la versione logaritmica può al massimo avvicinarsi allo stato dell'arte, ma non superarlo.
### Risultati di prova32bitMat
Inserisci un numero 1: 2131
Inserisci un numero 2: 3112
Inserisci la dimensione N delle matrici: 2048

Tempo impiegato standard:    0.898968 secondi
Tempo impiegato logaritmico: 1.167748 secondi
Rapporto standard/logaritmico: 0.8x
### Risultati prova128bitMat
./prova128bitMat
Inserisci un numero 1: 1233
Inserisci un numero 2: 1231
Inserisci la dimensione N delle matrici: 2048

Tempo impiegato standard:    178.998313 secondi
Tempo impiegato logaritmico: 87.351225 secondi
Rapporto standard/logaritmico: 2.0x
## Dove l'approccio logaritmico si dimostra efficace

Ho osservato che il metodo diventa vantaggioso in tutte le operazioni per cui la CPU non ha un'unità hardware veloce.
Il caso più interessante per noi, che vogliamo implementare la moltiplicaizone matriciale, è quello dei float a 128 bit. Le CPU x86 non hanno circuiti per questo formato, mentre l'acceleratore CPU si attiva per float a 32 bit oppure a 64(da testare fino a dove arriva, preso 128 bit a caso).
Va notato anche un limite importante: l'errore massimo resta l'11%, identico a quello dei 32 bit, perché dipende dall'approssimazione del logaritmo e non dal numero di bit.
Anche la divisione è un caso favorevole, perché in hardware è molto più lenta della moltiplicazione, mentre con l'approccio logaritmico diventa una sottrazione intera dei bit più il bias.
Altri contesti in cui l'idea ha senso, non ancora misurati, sono i microcontrollori che non hanno il floating point e sull'hardware dedicato come FPGA e ASIC, dove un sommatore occupa molta meno area rispetto a un moltiplicatore e consuma molta meno energia di un moltiplicatore.
