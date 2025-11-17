#include <stdio.h>

// Kernel definition
// __global__ void VecSdd(float* A, float* B, float* C) {
//     int i = threadIdx.x;
//     C[i] = A[i] + B[i];
// }

double add(double* a, double* b) {
    return *a + *b;
}

int main() {
    double A  = 2.00;
    double B  = 3.00;
    double x = add(&A, &B);
    // VecAdd<<<1, N>>>(A, B, C);   
    printf("Hello, World! :: %.2f\n", x);
}