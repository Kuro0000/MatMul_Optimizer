# MatMul_Optimizer
## Codice di base:
La base è la funzione nota come *fast inverse square root*, che calcola 1/√x lavorando direttamente sui bit del numero floating point.
```c
float Q_rsqrt( float number )
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
//	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}
```
## Idea
Sfruttare le caratteristiche dello standard IEEE 754, in cui i bit di un numero floating point, letti come intero, rappresentano già (a meno di una scala e di uno spostamento) il suo logaritmo in base 2. In questo modo si può usare la proprietà dei logaritmi per cui la moltiplicazione diventa una somma: invece di moltiplicare due numeri si sommano i loro bit come interi.
L'obiettivo del progetto è capire quando conviene davvero sostituire un'operazione floating point con il trucco dei bit, su CPU e su GPU, e perché

## Struttura del progetto

### Fondamenti
IEEE 754, i bit come log2, cosa diventa ogni operazione, errore al variare di K, casi limite.
### Studio CPU: moltiplicazione e moltiplicazione tra matrici(Modulo 1)
perde su CPU e GPU con formati previsti dalla FMA. FP128 è il caso "vince ma perde precisione", spiegando come abbiamo cercato di diminuire l'errore.
### Studio GPU: algoritmo da definire assieme(Modulo 2)
Con il supporto FMA nativo in ogni CUDA core, il trucco dei logaritmi è meno preciso e meno efficiente dell'approccio standard.
1. Moltiplicazione matriciale: con l'FMA in ogni CUDA core il trucco perde
2. Algoritmo principale in cui la SFU che opera log2 oppure exp2 è il collo di bottiglia.
3. Ipotesi: il trucco non accelera la SFU, ma le toglie lavoro spostandolo sulla pipeline intera. Le due unità lavorano in parallelo.
4. Le tappe implementative parte da una versione naive in CUDA, con progressivamente delle ottimizzazioni. 
  A ogni passo analisi con Nsight Compute;
### Conclusioni
Tabella conclusiva
