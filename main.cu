
#include <cuda_runtime.h>

#include "csvloader.h"

#define WINDOW      25.0f

#define DEBUG

__global__ void signalMagnitude(COLUMN_TYPE *ans, const COLUMN_TYPE *arr, int x, int y) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i + WINDOW < y) {
    int j, k, l;
    COLUMN_TYPE sig;
    for (j = 0; j < WINDOW; j++, l = i + j)
      for (k = 0; k < x; k++)
        sig += ABS_FUNC(arr[(l * x) + k]);
    ans[i] = sig / WINDOW;
  }
}

__global__ void averageMovementIntensity(COLUMN_TYPE *ans, const COLUMN_TYPE *arr, int x, int y) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i + WINDOW < y) {
    int j, k, l;
    COLUMN_TYPE sig = 0;
    for (j = 0; j < WINDOW; j++, l = i + j)
      for (k = 0; k < x; k++)
        sig += (arr[(l * x) + k] * arr[(l * x) + k]);
    ans[i] = sig / WINDOW;
  }
}

__global__ void standardDeviation(COLUMN_TYPE *w, COLUMN_TYPE*v, const COLUMN_TYPE *x, int numElements) {
  int i, k;
  k = i = blockDim.x * blockIdx.x + threadIdx.x;
  COLUMN_TYPE sum = 0, sig = 0, mean = 0;
  if (i + WINDOW < numElements) {
    for (int j = 0; j < WINDOW; j++, k++)
      sum += x[k];
    mean = sum / WINDOW;
    k = i;
    for (int j = 0; j < WINDOW; j++, k++)
      sig += x[k] - mean;
    sig *= sig;
    sig /= WINDOW;
    v[i] = mean;
    w[i] = SQRT_FUNC(sig);
  } else {
    w[i] = 0;
  }
}

int main(int argc, char **argv) {

  char *fn;
  int x, y;
  FILE *csv;
  COLUMN_TYPE *arr, *dev, *avg, *mag, *ami;

  fn = argv[1];

  csv = fopen(fn, "r");
  if (csv == NULL) {
    fprintf(stderr, "Failed to open file %s\n", fn);
    return EXIT_FAILURE;
  }

  x = y = 0;
  if (rowct(csv, &y) == EXIT_FAILURE || colct(csv, &x) == EXIT_FAILURE)
    return EXIT_FAILURE;
  if (readcsv(csv, x, y, &arr) == EXIT_FAILURE)
    return EXIT_FAILURE;
  fclose(csv);

  int threadsPerBlock = 128;
  int blocksPerGrid =(y + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

  COLUMN_TYPE *d_arr = NULL;
  COLUMN_TYPE *d_mag = NULL;
  COLUMN_TYPE *d_ami = NULL;
  COLUMN_TYPE *d_dev = NULL;
  COLUMN_TYPE *d_avg = NULL;

  size_t ct_size = sizeof(COLUMN_TYPE);

  cudaMalloc((void **)&d_arr, x * y * ct_size);
  cudaMalloc((void **)&d_mag, y * ct_size);
  cudaMalloc((void **)&d_ami, y * ct_size);
  cudaMalloc((void **)&d_dev, x * y * ct_size);
  //cudaMalloc((void **)&d_avg, x * y * ct_size);

  cudaMemcpy(d_arr, arr, y * ct_size, cudaMemcpyHostToDevice);

  signalMagnitude<<<blocksPerGrid, threadsPerBlock>>>(d_mag, arr, x, y);
  averageMovementIntensity<<<blocksPerGrid, threadsPerBlock>>>(d_ami, arr, x, y);

  mag = (COLUMN_TYPE *)calloc(y, ct_size);
  ami = (COLUMN_TYPE *)calloc(y, ct_size);
  dev = (COLUMN_TYPE *)calloc(x * y, ct_size);
  avg = (COLUMN_TYPE *)calloc(x * y, ct_size);

  cudaMemcpy(mag, d_mag, y * ct_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(ami, d_ami, y * ct_size, cudaMemcpyDeviceToHost);

  for (int q = 0; q < y; q++) {
    fprintf(stdout, "%d: %f %f\n", q, mag[q], ami[q]);
  }

  cudaFree(d_arr);
  cudaFree(d_mag);
  cudaFree(d_ami);
  cudaFree(d_dev);
  cudaFree(d_avg);

  free(arr);
  free(mag);
  free(ami);

  return 0;
}
