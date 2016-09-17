#include "const.h"
#include "kernels.h"

#define ABS_FUNC                    fabs
#define SQRT_FUNC                   sqrtf

__global__
void signalMagnitude(
  COLUMN_TYPE *ans, const COLUMN_TYPE *arr, int x, int y) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i + WINDOW < y) {
    int j, k;
    COLUMN_TYPE sig = 0;
    for (j = 0; j < WINDOW; j++) {
      for (k = 0; k < x; k++) {
        sig += ABS_FUNC(arr[(x * (i + j)) + k]);
      }
    }
    ans[i] = sig / WINDOW;
  } else if (i < y) {
    ans[i] = 0;
  }
}

__global__
void averageMovementIntensity(
  COLUMN_TYPE *ans, const COLUMN_TYPE *arr, int x, int y) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i + WINDOW < y) {
    int j, k;
    COLUMN_TYPE sig = 0;
    for (j = 0; j < WINDOW; j++) {
      for (k = 0; k < x; k++) {
        int p = (x * (i + j)) + k;
        sig += arr[p] * arr[p];
      }
    }
    ans[i] = sig / WINDOW;
  } else if (i < y) {
    ans[i] = 0;
  }
}

__global__
void standardDeviation(
  COLUMN_TYPE *dev, COLUMN_TYPE *avg, const COLUMN_TYPE *arr,
  int x, int y, int xy) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i + (WINDOW * x) < xy) {
    int j;
    COLUMN_TYPE mean, sig, sum;
    sum = 0; sig = 0;
    for (j = 0; j < WINDOW; j++)
      sum += arr[i + (j * x)];
    mean = sum / WINDOW;
    for (int j = 0; j < WINDOW; j++)
      sig += arr[i + (j * x)] - mean;
    sig *= sig;
    sig /= WINDOW;
    avg[i] = mean;
    dev[i] = SQRT_FUNC(sig);
  } else if (i < xy) {
    avg[i] = 0;
    dev[i] = 0;
  }
}
