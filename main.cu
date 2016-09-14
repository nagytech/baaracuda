
/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define NUM_ELEMENTS 50000

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // We might end up launching more threads than elements because
    // we launch in block-sized denominations.
    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main Program
 */
int
main(void)
{

    // Print the vector length to be used, and compute its size
    size_t size = NUM_ELEMENTS * sizeof(float);
    printf("[Vector addition of %d elements]\n", NUM_ELEMENTS);

    // Allocate the host array A
    float *h_A = (float *)malloc(size);

    // Allocate the host array B
    float *h_B = (float *)malloc(size);

    // Allocate the host  array C to store the result
    float *h_C = (float *)malloc(size);

    // Initialize arrays A and B
    for (int i = 0; i < NUM_ELEMENTS; ++i)
    {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Allocate the device array A
    float *d_A = NULL;
    cudaMalloc((void **)&d_A, size);

    // Allocate the device array B
    float *d_B = NULL;
    cudaMalloc((void **)&d_B, size);

    // Allocate the device array C
    float *d_C = NULL;
    cudaMalloc((void **)&d_C, size);

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy data from the host memory to the CUDA device\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch the VectorAdd CUDA Kernel
    int threadsPerBlock = 128;
    // Number of Blocks - ensures 1 thread be element
    int blocksPerGrid =(NUM_ELEMENTS + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, NUM_ELEMENTS);

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify that the result vector is correct
    for (int i = 0; i < NUM_ELEMENTS; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

	// Free device global memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free host memory
	free(h_A);
	free(h_B);
	free(h_C);

	return 0;
}

