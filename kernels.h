#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "const.h"

/* Threads per block */
#define TPB                           128

/* Public Functions */
__global__
void signalMagnitude(
  DATA_T *ans, const DATA_T *arr,
  int x, int y);

__global__
void averageMovementIntensity(
  DATA_T *ans, const DATA_T *arr,
  int x, int y);

__global__
void standardDeviation(
  DATA_T *dev, DATA_T *avg, const DATA_T *arr,
  int x, int y, int xy);

__global__
void minmax(
  DATA_T *min, DATA_T *max, const DATA_T *arr,
  int x, int y, int xy);

#endif /* _KERNELS_H_ */
