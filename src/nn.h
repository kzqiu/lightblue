#ifndef NN_H
#define NN_H

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

typedef double (*error_fn_t)(double *, double *, int);
typedef double (*activation_fn_t)(double, int);

// Deep Neural Network (Multilayer Perceptron)
typedef struct _layer {
    int n_neurons;
    int n_prev_neurons;
    double *weights; // n_neurons * n_prev_neurons = matrix of weights!
    double *biases; // n_neurons
    double *act_values; // n_neurons
    activation_fn_t activation;
} layer_t;

typedef struct _dnn {
    int n_layers;
    int input_size;
    layer_t *layers;
    error_fn_t error_fn;
} dnn_t;

dnn_t *dnn_new(int n_layers, int *sizes, int input_size, error_fn_t error_fn, activation_fn_t *activation);

void dnn_train(dnn_t *nn, int n_epochs, double learning_rate, int n_training_points, double *x, double *y);

void dnn_free(dnn_t *nn);

#endif