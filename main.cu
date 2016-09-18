
#include <cuda_runtime.h>

#include "const.h"
#include "runner.h"
#include "kernels.h"
#include "csvloader.h"

#define DEBUG

int main(int argc, char **argv) {

  char *fn;
  int x, y;
  DATA_T *data;
  DATA_T *mag, *ami, *dev, *avg;

  fn = argv[1];

  if (loadcsv(fn, data, &x, &y) != EXIT_SUCCESS) {
    fprintf(stderr, "Failed to load CSV file\n");
    return EXIT_FAILURE;
  }

  if (do_calcs(data, mag, ami, dev, avg) != EXIT_SUCCESS) {
    printf(stderr, "Failed to perform one or more calculations\n");
    if (data != NULL)
      free(data);
    return EXIT_FAILURE;
  }

  /* Output the results */
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

  /* Release all memory */
  free(arr);
  free(mag);
  free(ami);
  free(dev);
  free(avg);

  return EXIT_SUCCESS;

}
