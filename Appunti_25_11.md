# Progressi
GPU: Tesla T4 (40 SM, cc 7.5, clock nominale 1590 MHz)

=== ACCURATEZZA di 2^x per x in [-20, 0] ===
MUFU (ex2.approx)          errore relativo max 1.39e-07   medio 5.22e-08   medio con segno -5.1e-08
tabella (trucco dei bit)   errore relativo max 6.02e-04   medio 1.21e-04   medio con segno -6.9e-07
polinomio (come FA4)       errore relativo max 1.01e-04   medio 5.28e-05   medio con segno +4.4e-05

=== THROUGHPUT: softmax sintetica, 10.74 miliardi di exp2 per esecuzione ===
metodo       sw    tempo [ms]   Gexp2/s   exp2/ciclo/SM   vs MUFU   diff. somma
MUFU         0/8      22.438      478.5          7.5     1.00x   +0.0e+00
tabella      1/8      11.491      934.4         14.7     1.95x   -4.9e-08
tabella      2/8      11.719      916.3         14.4     1.91x   -1.1e-07
tabella      3/8      11.388      942.9         14.8     1.97x   -1.7e-07
tabella      4/8      12.046      891.3         14.0     1.86x   -2.3e-07
tabella      5/8      12.616      851.1         13.4     1.78x   -2.9e-07
tabella      6/8      13.619      788.4         12.4     1.65x   -3.5e-07
tabella      7/8      15.039      714.0         11.2     1.49x   -4.0e-07
tabella      8/8      16.262      660.3         10.4     1.38x   -4.5e-07
polinomio    1/8      12.455      862.1         13.6     1.80x   +5.5e-06
polinomio    2/8      13.106      819.3         12.9     1.71x   +1.1e-05
polinomio    3/8      15.757      681.4         10.7     1.42x   +1.7e-05
polinomio    4/8      18.041      595.2          9.4     1.24x   +2.2e-05
polinomio    5/8      20.066      535.1          8.4     1.12x   +2.8e-05
polinomio    6/8      22.575      475.6          7.5     0.99x   +3.3e-05
polinomio    7/8      25.006      429.4          6.8     0.90x   +3.9e-05
polinomio    8/8      27.535      390.0          6.1     0.81x   +4.4e-05

sw = esponenziali su 8 calcolati in software (gli altri sulla MUFU).
exp2/ciclo/SM e' calcolato al clock nominale: se sotto carico la GPU gira piu' bassa,
il valore vero e' piu' alto. Il limite teorico della sola MUFU e' 16 per ciclo per SM.
