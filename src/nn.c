#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "nn.h"
#include "util.h"
#include "blis.h"

// Perceptron Functions
perceptron_t *perceptron_new(int dimension) {
    if (dimension < 1) return NULL;
    perceptron_t *p = (perceptron_t *) malloc(sizeof(perceptron_t));
    p->weights = calloc(dimension, sizeof(double));
    p->bias = 0.0;
    p->dimension = dimension;
    return p;
}

void perceptron_free(perceptron_t *perceptron) {
    free(perceptron->weights);
    free(perceptron);
}

void perceptron_print(perceptron_t *perceptron) {
    printf("Bias = %f\nWeights:\n", perceptron->bias);
    for (int i = 0; i < perceptron->dimension; i++) {
        printf("%f ", *(perceptron->weights + i));
    }
    printf("\n");
}

void perceptron_train(perceptron_t *perceptron, int n_iter, int n_training_points, double *x, double *y) {
    for (int i = 0; i < n_iter; i++) {
        for (int j = 0; j < n_training_points; j++) {
            double a = _dot_product(perceptron->weights, x + j * perceptron->dimension, perceptron->dimension) + perceptron->bias;

            if (y[j] * a <= 0) {
                for (int k = 0; k < perceptron->dimension; k++) {
                    perceptron->weights[k] += y[j] * x[j * perceptron->dimension + k];
                }
                perceptron->bias += y[j];
            }
        }
    }
}

void perceptron_train_avg(perceptron_t *perceptron, int n_iter, int n_training_points, double *x, double *y) {
    double *cached_weights = calloc(perceptron->dimension, sizeof(double));
    double cached_bias = 0;
    int c = 1;

    for (int i = 0; i < n_iter; i++) {
        for (int j = 0; j < n_training_points; j++) {
            double a = _dot_product(perceptron->weights, x + j * perceptron->dimension, perceptron->dimension) + perceptron->bias;

            if (y[j] * a <= 0) {
                for (int k = 0; k < perceptron->dimension; k++) {
                    perceptron->weights[k] += y[j] * x[j * perceptron->dimension + k];
                    cached_weights[k] += y[j] * x[j * perceptron->dimension + k] * c;
                }
                perceptron->bias += y[j];
                cached_bias += y[j] * c;
            }
            c++;
        }
    }

    perceptron->bias -= 1.0 / c * cached_bias;

    for (int i = 0; i < perceptron->dimension; i++) {
        perceptron->weights[i] -= 1.0 / c * cached_weights[i];
    }

    free(cached_weights);
}

double perceptron_test(perceptron_t *perceptron, double *x) {
    return _dot_product(perceptron->weights, x, perceptron->dimension) + perceptron->bias;
}

// Deep Neural Network (Multilayer Perceptron)
dnn_t *dnn_new(int n_layers, int *sizes, int input_size, error_fn_t error_fn, activation_fn_t *activation) {
    double zero = 0;
    if (n_layers < 1 || !sizes) return NULL;
    dnn_t *nn = (dnn_t *) malloc(sizeof(dnn_t));
    nn->n_layers = n_layers;
    nn->layers = (layer_t *) malloc(n_layers * sizeof(layer_t));
    nn->error_fn = error_fn;

    // initialize each layer as well!
    for (int i = 0; i < n_layers; i++) {
        layer_t *l = &nn->layers[i];

        l->n_neurons = sizes[i];
        if (!i) l->n_prev_neurons = input_size;
        else l->n_prev_neurons = sizes[i - 1];
        l->activation = activation[i];

        // dynamic allocations
        l->weights = (double *) malloc(sizes[i] * l->n_prev_neurons * sizeof(double));
        l->biases = (double *) malloc(sizes[i] * sizeof(double));
        l->act_values = (double *) malloc(sizes[i] * sizeof(double));

        // activate the weights and biases to random values between -1 and 1, set activation values to 0
        bli_drandv(l->n_prev_neurons * sizes[i], l->weights, 1);
        bli_drandv(sizes[i], l->biases, 1);
        bli_dsetv(BLIS_NO_CONJUGATE, sizes[i], &zero, l->act_values, 1);
    }

    return nn;
}

void dnn_train(dnn_t *nn, int n_epochs, double learning_rate, int n_training_points, double *x, double *y) {
    double alpha = 1.0, beta = 0.0; // scaling factors required by matrix multiplication function from BLIS

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        for (int p = 0; p < n_training_points; p++) {
            // forward propagation
            double *prev = x + p * nn->layers[0].n_neurons;
            int output_size;

            for (int l = 0; l < nn->n_layers; l++) {
                layer_t layer = nn->layers[l];

                // W[l] * act[l - 1] = act[l] where W is the matrix of weights for layer l
                bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, layer.n_neurons, 1, layer.n_prev_neurons, &alpha, layer.weights, layer.n_prev_neurons, 1, prev, 1, 1, &beta, layer.act_values, 1, 1); 

                // add biases to vector
                bli_daddv(BLIS_NO_CONJUGATE, layer.n_neurons, layer.biases, 1, layer.act_values, 1);

                // activate all nodes in vector
                activate_all(layer.act_values, layer.n_neurons, layer.activation, 0);

                prev = layer.act_values;
                if (l == nn->n_layers - 1) output_size = layer.n_neurons;
            }

            // final output should be a column vector still of dimensions final layer n_neurons * 1, transformed by activation fn!
            // calculate error
            
            // nn->layers[nn->n_layers - 1].n_neurons == output size (written in a long way)
            nn->error_fn(y + output_size * p, prev, output_size);

            // TODO: backward propagation
            // keep track of activations actually! -> maybe we pre-allocate another vector for each layer?
            
        }
    }
}

void dnn_free(dnn_t *nn) {
    for (int i = 0; i < nn->n_layers; i++) {
        // cleaning up after dynamic allocations
        layer_t *l = nn->layers + i;
        free(l->weights);
        free(l->biases);
        free(l->act_values);
    }

    free(nn->layers);
    free(nn);
}
