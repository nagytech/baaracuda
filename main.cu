
#include <string>
#include <sstream>
#include <cuda_runtime.h>


#include "csvloader.h"

#define DEBUG

int main(int argc, char **argv)
{
  char *fn;
  int x, y;
  FILE *csv;
  COLUMN_TYPE **arr;

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

  if (readcsv(csv, x, y, &arr) == EXIT_FAILURE) {
    return EXIT_FAILURE;
  }

  fclose(csv);

  fprintf(stdout, "%f\n", arr[1][1]);

  return 0;
}
