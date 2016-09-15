
/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#define COLUMN_DELIMITER            ","
#define COLUMN_TYPE                 float
#define COLUMN_TYPE_FORMAT          "%f"
#define HEADER_ROWS                 0
#define MAX_COLUMNS                 5
#define MAX_LINE_BUFFER             1024
#define MAX_ROWS                    32 * MAX_COLUMNS * 1024

#define READ_CSV_OK                  0
#define ERR_MAX_COL_EXCEEDED        -1
#define ERR_MAX_COL_EXCEEDED_M      \
  "Maximum allowable column size exceeded.  Line %d has %d columns, maximum %d\n"
#define ERR_MAX_ROW_EXCEEDED_M      \
  "Maximum row count exceeded: found %d, maximum %d\n"
#define ERR_ROW_NOT_FOUND_M         \
  "Row %d could not be read\n"
#define ERR_COL_MISMATCH            -2
#define ERR_COL_MISMATCH_M          \
  "Column count mismatch. Line %d has %d columns, expected %d\n"
#define ERR_COL_PARSE_M             \
  "Column parsing error. line %d, column %d\n"

#define DEBUG

int rowct(FILE *fp, int *y);
int colct(FILE *fp, int *x);

int readcsv(FILE *fp, int x, int y, COLUMN_TYPE ***arr_out);

int readcsv(FILE *fp, int x, int y, COLUMN_TYPE ***arr_out) {

  int e, i;
  size_t lbmax;
  char *line, *colval;
  int rct, cct;
  COLUMN_TYPE **arr;

  e = READ_CSV_OK;

  if (fp == NULL) {
    fprintf(stderr, "File pointer was null.\n");
    return EXIT_FAILURE;
  }

  if (fseek(fp, 0, SEEK_SET) != 0) {
    fprintf(stderr, "Could not seek start of file.\n");
    return EXIT_FAILURE;
  }

  line = (char *)malloc(MAX_LINE_BUFFER * sizeof(char));
  if (line == NULL) {
    fprintf(stderr, "Out of memory.  Could not allocate for column counter\n");
    return EXIT_FAILURE;
  }

  arr = (COLUMN_TYPE **)malloc(y * sizeof(COLUMN_TYPE *) + (y * (x * sizeof(COLUMN_TYPE))));
  if (arr == NULL) {
    fprintf(stderr, "Out of memory.  Could not allocate for data array\n");
    return EXIT_FAILURE;
  }

  COLUMN_TYPE *data = (COLUMN_TYPE *)&arr[y];
  for (i = 0; i < y; i++, data += x)
      arr[i] = data;

  for (rct = 0; rct < y; rct++) {

    getline(&line, &lbmax, fp);

    if (rct < HEADER_ROWS) {
#ifdef DEBUG
      fprintf(stdout, "skipping header, row %d\n", rct);
#endif
      continue;
    }

    cct = 0;
    colval = strtok(line, COLUMN_DELIMITER);
    while (colval != NULL) {
#ifdef TRACE
      fprintf(stdout, "row %d:%d, value %s\n", rct, cct, colval);
#endif
      if (cct > MAX_COLUMNS) {
        e = ERR_MAX_COL_EXCEEDED;
        fprintf(stderr, ERR_MAX_COL_EXCEEDED_M, rct, cct, MAX_COLUMNS);
        break;
      }

      arr[rct][cct] = strtof(colval, NULL);

      cct++;
      colval = strtok(NULL, COLUMN_DELIMITER);
    }

    if (x != cct) {
      e = ERR_COL_MISMATCH;
      fprintf(stderr, ERR_COL_MISMATCH_M, rct, cct, x);
      break;
    }

  }

  if (e != READ_CSV_OK) {
    fprintf(stderr, ERR_COL_PARSE_M, rct, cct);
    // TODO: Free all lines.
    free(line);
    return EXIT_FAILURE;
  }

#ifdef DEBUG
  fprintf(stdout, "%d columns identified from csv file\n", cct);
#endif

  *arr_out = arr;

  return EXIT_SUCCESS;

}

int colct(FILE *fp, int *x) {

  int i, cols;
  char *line, *colval;
  size_t max_buf;

  if (fp == NULL) {
    fprintf(stderr, "File pointer was null.\n");
    return EXIT_FAILURE;
  }

  if (fseek(fp, 0, SEEK_SET) != 0) {
    fprintf(stderr, "Could not seek start of file.\n");
    return EXIT_FAILURE;
  }

  max_buf = MAX_LINE_BUFFER;
  line = (char *)calloc(max_buf, sizeof(char));
  if (line == NULL) {
    fprintf(stderr, "Out of memory.  Failed to assign buffer for line_count.\n");
    return EXIT_FAILURE;
  }

  for (i = 0; i < HEADER_ROWS + 1; i++) {
    getline(&line, &max_buf, fp);
  }

  if (line == NULL) {
    fprintf(stderr, ERR_ROW_NOT_FOUND_M, i);
    free(line);
    return EXIT_FAILURE;
  }

  cols = 0;
  colval = strtok(line, COLUMN_DELIMITER);
  while (colval != NULL) {
    cols++;
    colval = strtok(NULL, COLUMN_DELIMITER);
  }

  #ifdef DEBUG
    fprintf(stdout, "col count: %d\n", cols);
  #endif

  if (cols > MAX_COLUMNS) {
    fprintf(stderr, ERR_MAX_COL_EXCEEDED_M, cols, MAX_COLUMNS);
    free(line);
    return EXIT_FAILURE;
  }

  *x = cols;

  free(line);

  return EXIT_SUCCESS;

}


int rowct(FILE *fp, int *y) {

  int i;
  char *buf;
  size_t max_buf;

  if (fp == NULL) {
    fprintf(stderr, "File pointer was null.\n");
    return EXIT_FAILURE;
  }

  if (fseek(fp, 0, SEEK_SET) != 0) {
    fprintf(stderr, "Could not seek start of file.\n");
    return EXIT_FAILURE;
  }

  max_buf = MAX_LINE_BUFFER;
  buf = (char *)calloc(max_buf, sizeof(char));
  if (buf == NULL) {
    fprintf(stderr, "Out of memory.  Failed to assign buffer for line_count.\n");
    return EXIT_FAILURE;
  }

  i = 0;
  while (getline(&buf, &max_buf, fp) != EOF) {
    i++;
#ifdef DEBUG
    if (i % 10000 == 0)
    fprintf(stdout, "row count: %d...\n", i);
  #ifdef TRACE
    fprintf(stdout, "%d: %s", i, buf);
  #endif
#endif
  }

  if (i > MAX_ROWS) {
    fprintf(stderr, ERR_MAX_ROW_EXCEEDED_M, i, MAX_ROWS);
    free(buf);
    return EXIT_FAILURE;
  }

#ifdef DEBUG
  fprintf(stdout, "row count: %d\n", i);
#endif

  *y = i;

  free(buf);

  return EXIT_SUCCESS;

}

int main(int argc, char **argv)
{
  char *fn;
  COLUMN_TYPE **arr;
  FILE *csv;
  int x, y;

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
