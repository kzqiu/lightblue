#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "nn.h"

void perceptron_init(perceptron_t *perceptron, int dimension) {
    // initialize all weights to 0
    perceptron->weights = calloc(dimension, sizeof(float));
    perceptron->bias = 0.0;
    perceptron->dimension = dimension;
}

void perceptron_destruct(perceptron_t *perceptron) {
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

void perceptron_train(perceptron_t *perceptron, int n_iter, int n_training_points, float *x, float *y) {
    for (int i = 0; i < n_iter; i++) {
        for (int j = 0; j < n_training_points; j++) {
            float a = dot_product(perceptron->weights, x + j * perceptron->dimension, perceptron->dimension) + perceptron->bias;

            if (y[j] * a <= 0) {
                for (int k = 0; k < perceptron->dimension; k++) {
                    perceptron->weights[k] += y[j] * x[j * perceptron->dimension + k];
                }
                perceptron->bias += y[j];
            }
        }
    }
}

void perceptron_train_avg(perceptron_t *perceptron, int n_iter, int n_training_points, float *x, float *y) {
    float *cached_weights = calloc(perceptron->dimension, sizeof(float));
    float cached_bias = 0;
    int c = 1;

    for (int i = 0; i < n_iter; i++) {
        for (int j = 0; j < n_training_points; j++) {
            float a = dot_product(perceptron->weights, x + j * perceptron->dimension, perceptron->dimension) + perceptron->bias;

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

float perceptron_test(perceptron_t *perceptron, float *x) {
    return dot_product(perceptron->weights, x, perceptron->dimension) + perceptron->bias;
}

float dot_product(float *a, float *b, int size) {
    float sum = 0;

    for (int i = 0; i < size; i++) {
        sum += *(a + i) * *(b + i);
    }

    return sum;
}
