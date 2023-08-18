#ifndef NN_UTIL_H
#define NN_UTIL_H
#include <math.h>

// General utilities
static inline void activate_all(double *a, int size, double (*fn)(double, int), int use_derivative) {
    for (int i = 0; i < size; i++) a[i] = fn(a[i], use_derivative);
}

// Math functionality
static inline double _dot_product(double *a, double *b, int size) {
    double sum = 0;

    for (int i = 0; i < size; i++) {
        sum += *(a + i) * *(b + i);
    }

    return sum;
}

static inline double _sigmoid(double x, int use_derivative) {
    if (use_derivative) return x * (1 - x);
    return 1.0 / (1.0 + exp(x));
}

static inline double _tanh(double x, int use_derivative) {
    if (use_derivative) return 1 - pow(x, 2);
    return tanh(x);
}

static inline double _relu(double x, int use_derivative) {
    if (use_derivative) return x < 0 ? 0 : 1;
    return x < 0 ? 0 : x;
}

static inline double mse(double *a, double *b, int size) {
    double res = 0;
    for (int i = 0; i < size; i++) res += pow(a[i] - b[i], 2);
    return 0.5 * res;
}

#endif
