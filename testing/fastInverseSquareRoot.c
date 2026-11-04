#include <stdio.h>
#include <stdint.h>
#include <string.h>

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

float Q_log2( float number )
{
	int32_t i;
	float y = number;

	i = * ( int32_t * ) &y;                  // bit del float come intero

	return (float)i * (1.0f / 8388608.0f) - 126.95696f;   // i / 2^23 - (127 - σ)
}

float fast_log2_v2(float x) {
    uint32_t i;
    memcpy(&i, &x, sizeof i);

    float e = (float)((int)(i >> 23) - 127);          // esponente (parte intera del log)

    i = (i & 0x007FFFFFu) | 0x3F800000u;              // forza l'esponente a 0 → f in [1, 2)
    float f;
    memcpy(&f, &i, sizeof f);

    float t = f - 1.0f;                               // t = m'
    return e + t * (1.3465552f - 0.3465552f * t);     // ≈ log₂(1 + t)
}


int main()
{
    float x=2;
    printf("%f\n",Q_log2(2));
    printf("%f\n",fast_log2_v2(2));
    
    printf("%f", Q_rsqrt(2));

    return 0;
}
