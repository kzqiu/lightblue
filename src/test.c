#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

double input[4][2] = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1},
};
double and[4] = {-1, -1, -1, 1};
double or[4] = {-1, 1, 1, 1};
double nand[4] = {1, 1, 1, -1};

int main(int argc, char** argv) {
    perceptron_t *p = perceptron_new(2);
    perceptron_train_avg(p, 25, 4, input[0], or);
    perceptron_print(p);

    for (int i = 0; i < 4; i++) {
        double res = perceptron_test(p, input[i]);
        printf("(%f, %f) => %f\n", input[i][0], input[i][1], res);
    }

    perceptron_free(p);
    
    return 0;
}