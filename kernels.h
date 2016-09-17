#ifndef _KERNELS_H_
#define _KERNELS_H_

#include "const.h"

__global__
void signalMagnitude(
  COLUMN_TYPE *ans, const COLUMN_TYPE *arr,
  int x, int y);

__global__
void averageMovementIntensity(
  COLUMN_TYPE *ans, const COLUMN_TYPE *arr,
  int x, int y);

__global__
void standardDeviation(
  COLUMN_TYPE *dev, COLUMN_TYPE *avg, const COLUMN_TYPE *arr,
  int x, int y, int xy);

#endif /* _KERNELS_H_ */
