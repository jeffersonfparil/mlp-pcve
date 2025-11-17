#include <iostream.h>
#include <math.h>

void add(int n, float *x, float *y) {
    for (int i=0; i<n; i++) {
        y[i] = x[i] + y[i];
    }
}

int main(void) {
    int N = 1<<20; // 1 x 2^20 = 1,048,576
    float *x = new float[N];
    float *y = new float[N];
    for (int i=0; i<N; i++) {
        x[i] = i*1.0f;
        y[i] = i*2.0f;
    }
}