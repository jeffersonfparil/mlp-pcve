// cuda_wrapper.h

#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

// This guard ensures that if a C++ compiler (like nvcc) includes this,
// the function name is not mangled, making it linkable from C and Zig.
#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Performs vector addition on the GPU.
 * @param h_a Pointer to the first input vector on the host (CPU).
 * @param h_b Pointer to the second input vector on the host (CPU).
 * @param h_c Pointer to the output vector on the host (CPU), where the result is stored.
 * @param n The number of elements in each vector.
 */
void addVectorsCuda(const float *h_a, const float *h_b, float *h_c, int n);

#ifdef __cplusplus
}
#endif

#endif // CUDA_WRAPPER_H