#ifndef _CSVLOADER_H_
#define _CSVLOADER_H_

#include <stdio.h>
#include "const.h"

/* CSV Output Configuration */
#define DECIMAL_PLACES              8

int loadcsv(char *fn, DATA_T *data, int *x, int *y);


#endif /* _CSV_LOADER_H_ */
