#ifndef NN_H
#define NN_H

typedef struct perceptron {
    float *weights;
    float bias;
    int dimension;
} perceptron_t;

void perceptron_init(perceptron_t *perceptron, int dimension);
void perceptron_destruct(perceptron_t *perceptron);
void perceptron_print(perceptron_t *perceptron);
void perceptron_train(perceptron_t *perceptron, int n_iter, int n_training_points, float *x, float *y);
void perceptron_train_avg(perceptron_t *perceptron, int n_iter, int n_training_points, float *x, float *y);
float perceptron_test(perceptron_t *perceptron, float *x);
float dot_product(float *a, float *b, int size);

#endif