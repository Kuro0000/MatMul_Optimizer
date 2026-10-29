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
L'obiettivo del progetto è usare questa operazione per rendere più efficiente la moltiplicazione matriciale, e capire in quali condizioni e su quale hardware il trucco conviene davvero rispetto alla moltiplicazione standard.
