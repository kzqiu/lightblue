#include <stdio.h>
#include <stdlib.h>
#include "nn.h"
#include "blis.h"

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

    // Figuring out BLIS
    // int m = 2, n = 1, k = 3;
    // double zero = 0, one = 1;

    // double *a = malloc(m * k * sizeof(double));
    // double *b = malloc(k * n * sizeof(double));
    // double *c = malloc(m * n * sizeof(double));

    // a[0] = -1, a[1] = 3, a[2] = 4;
    // a[3] = 5, a[4] = 0, a[5] = 8;

    // bli_drandv(m * k, a, 1);
    // bli_drandm(0, BLIS_DENSE, m, k, a, k, 1);

    // bli_dprintm("a: ", m, k, a, 3, 1, "%4.1f", "");
    
    // b[0] = 3, b[1] = 4, b[2] = 10;

    // bli_dprintm("b: ", k, n, b, 1, 1, "%4.1f", "");

    // bli_dsetv(BLIS_NO_CONJUGATE, m * n, &zero, c, 1);
    // bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, m, n, k, &one, a, k, 1, b, n, 1, &zero, c, n, 1);

    // bli_dprintm("c: ", m, n, c, 1, 1, "%4.1f", "");

    
    // free(a);
    // free(b);
    // free(c);

    return 0;
}