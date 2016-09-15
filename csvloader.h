#ifndef _CSVLOADER_H_
#define _CSVLOADER_H_

#include <stdio.h>

/* Constants */
#define COLUMN_DELIMITER            ","
#define COLUMN_TYPE                 float
#define HEADER_ROWS                 0
#define MAX_COLUMNS                 8
#define MAX_LINE_BUFFER             1024
#define MAX_ROWS                    1024 * 1024

/* Error Codes, Messages */
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


int rowct(FILE *fp, int *y);
int colct(FILE *fp, int *x);

int readcsv(FILE *fp, int x, int y, COLUMN_TYPE ***arr_out, int *size);

#endif
