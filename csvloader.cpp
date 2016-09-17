/* csvloader.cpp
 * -------------
 * Facilitates the loading of data from a CSV file into memory.
 *
 */

#include <stdlib.h>
#include <string.h>
#include "csvloader.h"

/**
 * readcsv
 * -------
 * Reads the CSV file at `fp` into `arr_out` using row major order.
 *
 * @param  fp      file pointer to csv file
 * @param  x       width of data structure
 * @param  y       length of data structure
 * @param  arr_out output array
 *
 * @return         success or failure
 */
int readcsv(FILE *fp, int x, int y, COLUMN_TYPE **arr_out) {

  int e;
  size_t lbmax;
  char *line, *colval;
  int rct, cct, size;
  COLUMN_TYPE *arr;

  e = READ_CSV_OK;
  size = x * y;

  /* Check file exists */
  if (fp == NULL) {
    fprintf(stderr, "File pointer was null.\n");
    return EXIT_FAILURE;
  }

  /* Reset file index */
  if (fseek(fp, 0, SEEK_SET) != 0) {
    fprintf(stderr, "Could not seek start of file.\n");
    return EXIT_FAILURE;
  }

  /* Allocate for line buffer */
  line = (char *)malloc(MAX_LINE_BUFFER * sizeof(char));
  if (line == NULL) {
    fprintf(stderr, "Out of memory.  Could not allocate for column counter\n");
    return EXIT_FAILURE;
  }

  /* Allocate for output */
  arr = (COLUMN_TYPE *)calloc(size, sizeof(COLUMN_TYPE));
  if (arr == NULL) {
    fprintf(stderr, "Out of memory.  Could not allocate for data array\n");
    return EXIT_FAILURE;
  }

  /* Iterate the rows */
  for (rct = 0; rct < y; rct++) {
    getline(&line, &lbmax, fp);

    /* Check for and skip header rows if configured */
    if (rct < HEADER_ROWS) {
#ifdef DEBUG
      fprintf(stdout, "skipping header, row %d\n", rct);
#endif
      continue;
    }

    /* Iterate the columns within the current row */
    cct = 0;
    colval = strtok(line, COLUMN_DELIMITER);
    while (colval != NULL) {
#ifdef TRACE
      fprintf(stdout, "row %d:%d, value %s\n", rct, cct, colval);
#endif
      /* Check if this row exeeds the constraints */
      if (cct > MAX_COLUMNS) {
        e = ERR_MAX_COL_EXCEEDED;
        fprintf(stderr, ERR_MAX_COL_EXCEEDED_M, rct, cct, MAX_COLUMNS);
        break;
      }

      /* Apply vector indexing via row major order */
      arr[(rct * x) + cct] = strtof(colval, NULL);

      cct++;
      colval = strtok(NULL, COLUMN_DELIMITER);
    }

    if (x != cct) {
      e = ERR_COL_MISMATCH;
      fprintf(stderr, ERR_COL_MISMATCH_M, rct, cct, x);
      break;
    }

  }

  /* Check for success or failure */
  if (e != READ_CSV_OK) {
    fprintf(stderr, ERR_COL_PARSE_M, rct, cct);
    free(arr);
    free(line);
    return EXIT_FAILURE;
  }

  *arr_out = arr;

  return EXIT_SUCCESS;

}

/**
 * colct
 * -----
 * count the columns within the given input file
 *
 * @param  fp file pointer to input file
 * @param  x  width of the file, columns (output)
 *
 * @return    success or failure
 */
int colct(FILE *fp, int *x) {

  int i, cols;
  char *line, *colval;
  size_t max_buf;

  /* Check the file is available */
  if (fp == NULL) {
    fprintf(stderr, "File pointer was null.\n");
    return EXIT_FAILURE;
  }

  /* Reset file index */
  if (fseek(fp, 0, SEEK_SET) != 0) {
    fprintf(stderr, "Could not seek start of file.\n");
    return EXIT_FAILURE;
  }

  /* Allocate for the line buffer */
  max_buf = MAX_LINE_BUFFER;
  line = (char *)calloc(max_buf, sizeof(char));
  if (line == NULL) {
    fprintf(stderr, "Out of memory.  Failed to assign buffer for line_count.\n");
    return EXIT_FAILURE;
  }

  /* Check for and skip any headers if configured */
  for (i = 0; i < HEADER_ROWS + 1; i++) {
    getline(&line, &max_buf, fp);
  }

  /* Check that we still have a row */
  if (line == NULL) {
    fprintf(stderr, ERR_ROW_NOT_FOUND_M, i);
    free(line);
    return EXIT_FAILURE;
  }

  /* Split the row by the delimiter and count */
  cols = 0;
  colval = strtok(line, COLUMN_DELIMITER);
  while (colval != NULL) {
    cols++;
    colval = strtok(NULL, COLUMN_DELIMITER);
  }

  #ifdef DEBUG
    fprintf(stdout, "col count: %d\n", cols);
  #endif

  /* Check for constraints */
  if (cols > MAX_COLUMNS) {
    fprintf(stderr, ERR_MAX_COL_EXCEEDED_M, i, cols, MAX_COLUMNS);
    free(line);
    return EXIT_FAILURE;
  }

  *x = cols;

  free(line);

  return EXIT_SUCCESS;

}

/**
 * rowct
 * -----
 * count the number of rows within the given input file
 *
 * @param  fp file pointer for input file
 * @param  y  length of file in lines (excluding HEADER_ROWS)
 *
 * @return    success or failure
 */
int rowct(FILE *fp, int *y) {

  int i;
  char *buf;
  size_t max_buf;

  /* Check the file pointer is available */
  if (fp == NULL) {
    fprintf(stderr, "File pointer was null.\n");
    return EXIT_FAILURE;
  }

  /* Reset the file index to the beginning */
  if (fseek(fp, 0, SEEK_SET) != 0) {
    fprintf(stderr, "Could not seek start of file.\n");
    return EXIT_FAILURE;
  }

  /* Allocate for the line buffer */
  max_buf = MAX_LINE_BUFFER;
  buf = (char *)calloc(max_buf, sizeof(char));
  if (buf == NULL) {
    fprintf(stderr, "Out of memory.  Failed to assign buffer for line_count.\n");
    return EXIT_FAILURE;
  }

  /* Iterate each line of the input file and count (ignoring headers) */
  i = -HEADER_ROWS;
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

  /* Check for constraints */
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
