
#include <cuda_runtime.h>

#include "const.h"
#include "kernels.h"
#include "csvloader.h"

#define DEBUG

#define OUT_FORMAT_READING    ",%.0f"
#define OUT_FORMAT_MAG        ",%.0f"
#define OUT_FORMAT_AMI        ",%.0f"
#define OUT_FORMAT_AVG        ",%0.2f"
#define OUT_FORMAT_STD        ",%0.8f"


int main(int argc, char **argv) {

  char *fn;
  int x, y;
  FILE *csv;
  COLUMN_TYPE *arr;

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

  int tpb = 128;
  int bpg_multi =(y + tpb - 1) / tpb;
  int bpg_singl =((x * y) + tpb - 1) / tpb;

  size_t ct_size = sizeof(COLUMN_TYPE);

  COLUMN_TYPE *d_arr = NULL;
  cudaMalloc((void **)&d_arr, x * y * ct_size);
  cudaMemcpy(d_arr, arr, x * y * ct_size, cudaMemcpyHostToDevice);

  COLUMN_TYPE *d_mag = NULL, *mag;
  cudaMalloc((void **)&d_mag, y * ct_size);
  signalMagnitude<<<bpg_multi, tpb>>>(d_mag, d_arr, x, y);
  mag = (COLUMN_TYPE *)calloc(y, ct_size);
  cudaMemcpy(mag, d_mag, y * ct_size, cudaMemcpyDeviceToHost);
  cudaFree(d_mag);

  COLUMN_TYPE *d_ami = NULL, *ami = NULL;
  cudaMalloc((void **)&d_ami, y * ct_size);
  averageMovementIntensity<<<bpg_multi, tpb>>>(d_ami, d_arr, x, y);
  ami = (COLUMN_TYPE *)calloc(y, ct_size);
  cudaMemcpy(ami, d_ami, y * ct_size, cudaMemcpyDeviceToHost);
  cudaFree(d_ami);

  COLUMN_TYPE *d_dev = NULL, *d_avg = NULL, *dev = NULL, *avg = NULL;
  cudaMalloc((void **)&d_dev, x * y * ct_size);
  cudaMalloc((void **)&d_avg, x * y * ct_size);
  standardDeviation<<<bpg_singl, tpb>>>(d_dev, d_avg, d_arr, x, y, x * y);
  dev = (COLUMN_TYPE *)calloc(x * y, ct_size);
  avg = (COLUMN_TYPE *)calloc(x * y, ct_size);
  cudaMemcpy(dev, d_dev, x * y * ct_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(avg, d_avg, x * y * ct_size, cudaMemcpyDeviceToHost);
  cudaFree(d_dev);
  cudaFree(d_avg);

  fprintf(stdout, "ID");
  for (int r = 1; r <= x; r++)
    fprintf(stdout, ",INPUT_%d", r);
  fprintf(stdout, ",MAG,AMI");
  for (int r = 1; r <= x; r++)
    fprintf(stdout, ",STDEV_%d", r);
  for (int r = 1; r <= x; r++)
    fprintf(stdout, ",MEAN_%d", r);
  fprintf(stdout, "\n");
  for (int q = 0; q < y - WINDOW; q++) {
    fprintf(stdout, "%d", q);
    for (int r = 0; r < x; r++)
      fprintf(stdout, OUT_FORMAT_READING, arr[(q * x) + r]);
    fprintf(stdout, OUT_FORMAT_MAG, mag[q]);
    fprintf(stdout, OUT_FORMAT_AMI, ami[q]);
    for (int r = 0; r < x; r++)
      fprintf(stdout, OUT_FORMAT_STD, dev[(q * x) + r]);
    for (int r = 0; r < x; r++)
      fprintf(stdout, OUT_FORMAT_AVG, avg[(q * x) + r]);
    fprintf(stdout, "\n");
  }

  cudaFree(d_arr);
  cudaFree(d_mag);
  cudaFree(d_ami);
  cudaFree(d_dev);
  cudaFree(d_avg);

  free(arr);
  free(mag);
  free(ami);
  free(dev);
  free(avg);

  return 0;

}
