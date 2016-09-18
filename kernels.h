#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "const.h"

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

#endif /* _KERNELS_H_ */
