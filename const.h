#ifndef _CONST_H_
#define _CONST_H_

#define COLUMN_DELIMITER            ","
#define COLUMN_TYPE                 float
#define HEADER_ROWS                 0
#define MAX_COLUMNS                 8
#define MAX_LINE_BUFFER             1024
/* TODO: should be constrained by memory, or max size of int. But could also use long for indexer */
#define MAX_ROWS                    1024 * 1024
#define WINDOW                      25

#endif /* _CONST_H_ */
