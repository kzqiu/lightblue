#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "nn.h"
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
dnn_t *dnn_new(int n_layers, int *sizes) {
    if (n_layers < 1 || !sizes) return NULL;
    dnn_t *nn = (dnn_t *) malloc(sizeof(dnn_t));
    nn->n_layers = n_layers;
    return nn;
}

void dnn_train(dnn_t *nn, int n_epochs, int n_training_points, double *x, double *y) {
    for (int epoch = 0; epoch < n_epochs; epoch++) {
        for (int p = 0; p < n_training_points; p++) {
            // forward prop

            for (int l = 0; l < nn->n_layers; l++) {
                double *tmp = (double *) malloc(nn->layers[l].size * sizeof(double));
            }


            // calculate error


            // backward prop

        }
    }
}

void dnn_free(dnn_t *nn) {
    for (int i = 0; i < nn->n_layers; i++) {
        free(nn->layers + i);
    }
    free(nn);
}
