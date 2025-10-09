// kernel.cu

#include <stdio.h>

// A simple macro to check for CUDA errors
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

/**
 * @brief This is the CUDA kernel that runs on the GPU.
 * Each thread executes this function to add one element from each vector.
 */
__global__ void addVectorsKernel(const float *a, const float *b, float *c, int n) {
    // Calculate the global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread ID is within the bounds of the array
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief This is the C-callable wrapper function that runs on the CPU.
 * Zig will call this function.
 * It uses extern "C" to prevent C++ name mangling.
 */
extern "C" void addVectorsCuda(const float *h_a, const float *h_b, float *h_c, int n) {
    // Pointers for device (GPU) memory
    float *d_a, *d_b, *d_c;
    size_t size = n * sizeof(float);

    // 1. Allocate memory on the device
    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));

    // 2. Copy data from host (CPU) to device (GPU)
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // 3. Launch the kernel on the device
    // We define the grid and block dimensions for the kernel launch.
    int threadsPerBlock = 256;
    // Calculate the number of blocks needed in the grid
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    addVectorsKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // Check for any errors during kernel execution
    CHECK_CUDA(cudaGetLastError());

    // 4. Copy the result from device (GPU) back to host (CPU)
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // 5. Free the allocated device memory
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
}