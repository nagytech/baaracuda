/*
 * const.h
 * -------
 *
 * Author: Jonathan Nagy <jnagy@myune.edu.au>
 * Date:   18 Sep 2016
 *
 * Description:
 *
 *    Globally applicable constants for handling the csv and data processing
 *
 * ------------------------------------------------------------------------ */

#ifndef _CONST_H_
#define _CONST_H_

/* Configuration Values */
#define COLUMN_DELIMITER            ","           /* Input delimiter */
#define DATA_T                      float         /* Data type */
#define HEADER_ROWS                 0             /* Number of header rows */
#define MAX_COLUMNS                 8             /* Maximum CSV columns */
#define MAX_LINE_BUFFER             1024          /* Initially, will expand */
#define MAX_ROWS                    1024 * 1024   /* Max number of rows */
#define OUT_FORMAT_READING          ",%.0f"
#define OUT_FORMAT_MAG              ",%.0f"
#define OUT_FORMAT_AMI              ",%.0f"
#define OUT_FORMAT_AVG              ",%0.2f"
#define OUT_FORMAT_STD              ",%0.8f"
#define OUT_FORMAT_MIN              ",%.0f"
#define OUT_FORMAT_MAX              ",%.0f"
#define WINDOW                      25            /* Sliding window width */

/*
 * NOTE: For MAX_COLUMNS, MAX_ROWS we may want to constrain by several
 * system specifications (ie. 32/64 bit) for both memory and indexing purposes.
 * To do so would require a bit more customization of data types.
 */

#endif /* _CONST_H_ */
