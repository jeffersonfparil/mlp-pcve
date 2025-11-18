#include <stdio.h>

// Kernel definition
// __global__ void VecSdd(float* A, float* B, float* C) {
//     int i = threadIdx.x;
//     C[i] = A[i] + B[i];
// }

__global__
void cuda_hello() {
    printf("Hello from CUDA Kernel!\n");
}


double add(double* a, double* b) {
    return *a + *b;
}


int main() {
    double A  = 2.00;
    double B  = 3.00;
    double x = add(&A, &B);
    // VecAdd<<<1, N>>>(A, B, C);   
    // printf("Hello, World! :: %.2f\n", x);
    // cuda_hello<<<1,1>>>();
    // return 0;
}