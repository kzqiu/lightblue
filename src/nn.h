#ifndef NN_H
#define NN_H
#include <math.h>

// Math functionality
static inline double _dot_product(double *a, double *b, int size) {
    double sum = 0;

    for (int i = 0; i < size; i++) {
        sum += *(a + i) * *(b + i);
    }

    return sum;
}

static inline double _sigmoid(double x) {
    return 1.0 / (1.0 + exp(x));
}

static inline double _sigmoid_derivative(double x) {
    return x * (1 - x);
}

// double tanh(double x);

static inline double _tanh_derivative(double x) {
    return 1 - pow(x, 2);
}

static inline double _relu(double x) {
    return x < 0 ? 0 : x;
}

static inline double _relu_derivative(double x) {
    return x < 0 ? 0 : 1;
}

// Perceptron
typedef struct perceptron {
    double *weights;
    double bias;
    int dimension;
} perceptron_t;

perceptron_t *perceptron_new(int dimension);

void perceptron_free(perceptron_t *perceptron);

void perceptron_print(perceptron_t *perceptron);

void perceptron_train(perceptron_t *perceptron, int n_iter, int n_training_points, double *x, double *y);

void perceptron_train_avg(perceptron_t *perceptron, int n_iter, int n_training_points, double *x, double *y);

double perceptron_test(perceptron_t *perceptron, double *x);

// Deep Neural Network (Multilayer Perceptron)
typedef struct layer {
    int size;
    double *weights;
    double (*activation)(double);
} layer_t;

typedef struct dnn {
    int n_layers;
    layer_t *layers;
} dnn_t;

dnn_t *dnn_new(int n_layers, int *sizes);

void dnn_train(dnn_t *nn, int n_epochs, int n_training_points, double *x, double *y);

void dnn_free(dnn_t *nn);

#endif