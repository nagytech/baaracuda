
#include <cuda_runtime.h>

#include "unistd.h"
#include "csvloader.h"

#define DEBUG

__global__ void vectorAdd(float *w, const float *x, const float *y, const float *z, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        w[i] = x[i] + y[i] + z[i];
    }
}

int main(int argc, char **argv) {

  char *fn;
  int x, y, size;
  FILE *csv;
  COLUMN_TYPE **arr, *ans;

  fn = argv[1];

  csv = fopen(fn, "r");
  if (csv == NULL) {
    fprintf(stderr, "Failed to open file %s\n", fn);
    return EXIT_FAILURE;
  }

  x = y = 0;
  if (rowct(csv, &y) == EXIT_FAILURE || colct(csv, &x) == EXIT_FAILURE) {
    return EXIT_FAILURE;
  }
  if (readcsv(csv, x, y, &arr, &size) == EXIT_FAILURE) {
    return EXIT_FAILURE;
  }
  fclose(csv);

  int threadsPerBlock = 128;
  int blocksPerGrid =(y + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

  size_t size2 = y * sizeof(float);

  /* TODO: Not sure if we need to split it up this way */

  COLUMN_TYPE *d_w = NULL;
  COLUMN_TYPE *d_x = NULL;
  COLUMN_TYPE *d_y = NULL;
  COLUMN_TYPE *d_z = NULL;

  cudaMalloc((void **)&d_w, size2);
  cudaMalloc((void **)&d_x, size2);
  cudaMalloc((void **)&d_y, size2);
  cudaMalloc((void **)&d_z, size2);

  cudaMemcpy(d_x, arr[0], size2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, arr[1], size2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_z, arr[2], size2, cudaMemcpyHostToDevice);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_w, d_x, d_y, d_z, y);

  ans = (COLUMN_TYPE *)malloc(size2);

  cudaMemcpy(ans, d_w, y, cudaMemcpyDeviceToHost);

  for (int q = 0; q < y; q++) {
    float comp = arr[0][q] + arr[1][q] + arr[2][q];
    if (ans[q] > 0 && q % 1000 == 0)
    fprintf(stdout, "%d: %f %f\n", q, comp, ans[q]);
  }

  cudaFree(d_w);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  free(arr);
  free(ans);

  //fprintf(stdout, "%f\n", arr[2][0]);

  return 0;
}
