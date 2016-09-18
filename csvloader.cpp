/* csvloader.cpp
 * -------------
 * Facilitates the loading of data from a CSV file into memory.
 *
 */

#include <stdlib.h>
#include <string.h>
#include "csvloader.h"

/* Coded Errors */
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

/* Error Messages */
#define ERR_COL_PARSE_M             \
  "Column parsing error. line %d, column %d\n"
#define ERR_FP_NULL_M               "Failed to open file, NULL file pointer.\n"
#define ERR_OOM_DATA_ARRAY          \
  "Out of memory.  Failed to assign for data array.\n"
#define ERR_OOM_LINE_BUFFER_M       \
  "Out of memory.  Failed to assign buffer for line_count.\n"
#define ERR_SEEK_START_M            "Could not seek start of file.\n"
#define ERR_USAGE_M                 \
  "Error: no file name supplied.\n\tUsage %s <input_csv_filename>\n"

/* Debug messages */
#define DEBUG_LINE_COUNT            "col count: %d\n"
#define DEBUG_ROW_COUNT             "row count: %d\n"
#define DEBUG_ROW_COUNTING          "row count: %d...\n"
#define DEBUG_SKIP_HEADER           "skipping header, row %d\n"

/* Trace Messages */
#define TRACE_COLUMN_VALUE          "row %d:%d, value %s\n"
#define TRACE_LINE_BUFFER           "%d: %s\n"

int colct(FILE *fp, int *x);
int readcsv(FILE *fp, int x, int y, DATA_T **data);
int rowct(FILE *fp, int *y);


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
    fprintf(stderr, ERR_FP_NULL_M);
    return EXIT_FAILURE;
  }

  /* Reset file index */
  if (fseek(fp, 0, SEEK_SET) != 0) {
    fprintf(stderr, ERR_SEEK_START_M);
    return EXIT_FAILURE;
  }

  /* Allocate for the line buffer */
  max_buf = MAX_LINE_BUFFER;
  line = (char *)calloc(max_buf, sizeof(char));
  if (line == NULL) {
    fprintf(stderr, ERR_OOM_LINE_BUFFER_M);
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
    fprintf(stdout, DEBUG_LINE_COUNT, cols);
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
 * loadcsv
 * -------
 * handles the opening and reading of a csv file into memory including the
 * assignment of the data dimensions
 *
 * @param  fn   file name for input file
 * @param  data output of data (row major order)
 * @param  x    width of file (columns)
 * @param  y    length of file (lines, not including header)
 *
 * @return      success or failure
 */
int loadcsv(char *fn, DATA_T *data, int *x, int *y) {

  FILE *csv;

  /* Check file name is available */
  if (fn == NULL) {
    fprintf(stderr, ERR_USAGE_M, fn);
    return EXIT_FAILURE;
  }

  /* Open file */
  csv = fopen(fn, "r");
  if (csv == NULL) {
    fprintf(stderr, ERR_FP_NULL_M);
    return EXIT_FAILURE;
  }

  /* Check dimensions of the file */
  if (rowct(csv, y) == EXIT_FAILURE || colct(csv, x) == EXIT_FAILURE)
    return EXIT_FAILURE;

  /* Read CSV file data into memory */
  if (readcsv(csv, *x, *y, &data) == EXIT_FAILURE)
    return EXIT_FAILURE;

  fclose(csv);

  return EXIT_SUCCESS;

}

/**
 * readcsv
 * -------
 * Reads the CSV file at `fp` into `data_out` using row major order.
 *
 * @param  fp       file pointer to csv file
 * @param  x        width of data structure
 * @param  y        length of data structure
 * @param  data_out output array
 *
 * @return         success or failure
 */
int readcsv(FILE *fp, int x, int y, DATA_T **data_out) {

  int e;
  size_t lbmax;
  char *line, *colval;
  int rct, cct, size;
  DATA_T *data;

  e = READ_CSV_OK;
  size = x * y;

  /* Check file exists */
  if (fp == NULL) {
    fprintf(stderr, ERR_FP_NULL_M);
    return EXIT_FAILURE;
  }

  /* Reset file index */
  if (fseek(fp, 0, SEEK_SET) != 0) {
    fprintf(stderr, ERR_SEEK_START_M);
    return EXIT_FAILURE;
  }

  /* Allocate for line buffer */
  line = (char *)malloc(MAX_LINE_BUFFER * sizeof(char));
  if (line == NULL) {
    fprintf(stderr, ERR_OOM_LINE_BUFFER_M);
    return EXIT_FAILURE;
  }

  /* Allocate for output */
  data = (DATA_T *)calloc(size, sizeof(DATA_T));
  if (data == NULL) {
    fprintf(stderr, ERR_OOM_DATA_ARRAY);
    return EXIT_FAILURE;
  }

  /* Iterate the rows */
  for (rct = 0; rct < y; rct++) {
    getline(&line, &lbmax, fp);

    /* Check for and skip header rows if configured */
    if (rct < HEADER_ROWS) {
#ifdef DEBUG
      fprintf(stdout, DEBUG_SKIP_HEADER, rct);
#endif
      continue;
    }

    /* Iterate the columns within the current row */
    cct = 0;
    colval = strtok(line, COLUMN_DELIMITER);
    while (colval != NULL) {
#ifdef TRACE
      fprintf(stdout, TRACE_COLUMN_VALUE, rct, cct, colval);
#endif
      /* Check if this row exeeds the constraints */
      if (cct > MAX_COLUMNS) {
        e = ERR_MAX_COL_EXCEEDED;
        fprintf(stderr, ERR_MAX_COL_EXCEEDED_M, rct, cct, MAX_COLUMNS);
        break;
      }

      /* Apply vector indexing via row major order */
      data[(rct * x) + cct] = strtof(colval, NULL);

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
    free(data);
    free(line);
    return EXIT_FAILURE;
  }

  *data_out = data;

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
    fprintf(stderr, ERR_FP_NULL_M);
    return EXIT_FAILURE;
  }

  /* Reset the file index to the beginning */
  if (fseek(fp, 0, SEEK_SET) != 0) {
    fprintf(stderr, ERR_SEEK_START_M);
    return EXIT_FAILURE;
  }

  /* Allocate for the line buffer */
  max_buf = MAX_LINE_BUFFER;
  buf = (char *)calloc(max_buf, sizeof(char));
  if (buf == NULL) {
    fprintf(stderr, ERR_OOM_LINE_BUFFER_M);
    return EXIT_FAILURE;
  }

  /* Iterate each line of the input file and count (ignoring headers) */
  i = -HEADER_ROWS;
  while (getline(&buf, &max_buf, fp) != EOF) {
    i++;
#ifdef DEBUG
    if (i % 10000 == 0)
    fprintf(stdout, DEBUG_ROW_COUNTING, i);
  #ifdef TRACE
    fprintf(stdout, TRACE_LINE_BUFFER, i, buf);
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
  fprintf(stdout, DEBUG_ROW_COUNT, i);
#endif

  *y = i;

  free(buf);

  return EXIT_SUCCESS;

}
