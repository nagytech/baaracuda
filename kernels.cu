/*
 * kernels.cu
 * ----------
 *
 * Author: Jonathan Nagy <jnagy@myune.edu.au>
 * Date:   18 Sep 2016
 * Description:
 *
 *    This file contains a set of CUDA kernels used for computing various
 * statistical properties.  Note that all values are calculated within a
 * sliding window of `n` to `n + (WINDOW - 1)`.
 *
 *    All kernels are capable of accepting any number of input values and will
 * collapse the output into a single value.  However, some kernels are designed
 * to compute values _across_ features, while others are designed to compute
 * _within_ features.  This is determined by the presence of a precalculated
 * size (written as xy) which depticts the complete dimensions of the array and
 * indicates that each individual feature is to be used as an input.
 *
 * ------------------------------------------------------------------------ */

#include "const.h"
#include "kernels.h"

/* Function definitions */
#define ABS_FUNC                    fabs    /* For computing absolute value */
#define SQRT_FUNC                   sqrtf   /* For computing square root */

/*
 * signalMagnitude
 * ---------------
 * Calculates the signal magnitude across all input features
 *
 * Math TeX 3 feature example:
 *
 *   SMA =
 *     \frac{1}{T}\big(
 *       \sum\limits_{i=1}^T |a_x(i)| +
 *       \sum\limits_{i=1}^T |a_y(i)| +
 *       \sum\limits_{i=1}^T |a_z(i)|
 *     \big)
 *
 * ans:   array of output values, will be of length y
 * arr:   array of input values, stored in row major order by rank (x, y)
 * x:     width of data array
 * y:     height of data array
 *
 */
__global__
void signalMagnitude(
  DATA_T *ans, const DATA_T *arr, int x, int y) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  /* Filter over-allocated threads, but ignore last (window - 1) records */
  if (i + WINDOW < y) {
    int j, k;
    DATA_T sig = 0;
    /* Iterate through sliding window */
    for (j = 0; j < WINDOW; j++) {
      /* Iterate through columns within current window position */
      for (k = 0; k < x; k++) {
        /* Accumulate the signal strength */
        sig += ABS_FUNC(arr[(x * (i + j)) + k]);
      }
    }
    /* Mean all values */
    ans[i] = sig / WINDOW;
  } else if (i < y) {
    /* Avoid a NULL reference */
    ans[i] = 0;
  }
}

/*
 * averageMovementIntensity
 * ------------------------
 * Calculates the average movement intensity across all input features
 *
 * Math TeX 3 feature example:
 *
 *   MI_{avg} = \frac{1}{T}\big(
 *      \sum\limits_{i=1}^T (
 *        a_x(i)^2 + a_y(i)^2)+a_z(i)^2
 *      )
 *   \big)
 *
 * ans:   array of output values, will be of length y
 * arr:   array of input values, stored in row major order by rank (x, y)
 * x:     width of data array
 * y:     height of data array
 *
 */
__global__
void averageMovementIntensity(
  DATA_T *ans, const DATA_T *arr, int x, int y) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  /* Filter over-allocated threads, but ignore last (window - 1) records */
  if (i + WINDOW < y) {
    int j, k;
    DATA_T sig = 0;
    /* Iterate through sliding window */
    for (j = 0; j < WINDOW; j++) {
      /* Iterate through columns within current window position */
      for (k = 0; k < x; k++) {
        /* Accumulate the intensity value */
        int p = (x * (i + j)) + k;
        sig += arr[p] * arr[p];
      }
    }
    /* Mean all values */
    ans[i] = sig / WINDOW;
  } else if (i < y) {
    /* Avoid a null reference */
    ans[i] = 0;
  }
}

/*
 * standardDeviation
 * -----------------
 * Calculates the standard deviaion _and_ mean _for each_ input feature
 * across all input features.
 *
 * Math TeX example:
 *
 * sd_x = \sqrt{\sum\limits_{i=1}^T (a_x(i) - \bar{a_x})^2}
 * ax_{average} = \frac{1}{T}\big(\sum\limits_{i=1}^T (a_x(i) \big)
 *
 * dev:   array of output values, will be of length (y * x) and indexed by
          row major order rank (x, y) - (standard deviation)
 * avg:   array of output values, will be of length (y * x) and indexed by
          row major order rank (x, y) - (mean)
 * arr:   array of input values, stored in row major order by rank (x, y)
 * x:     width of data array
 * y:     height of data array
 *
 */
__global__
void standardDeviation(
  DATA_T *dev, DATA_T *avg, const DATA_T *arr,
  int x, int y, int xy) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  /* Filter over-allocated threads, but ignore last (window - 1) records */
  if (i + (WINDOW * x) < xy) {
    int j;
    DATA_T mean, sig, sum;
    sum = 0; sig = 0;
    /* Iterate through sliding window for summation */
    for (j = 0; j < WINDOW; j++)
      sum += arr[i + (j * x)];
    mean = sum / WINDOW;
    /* Iterate through sliding for standard deviation */
    for (int j = 0; j < WINDOW; j++)
      sig += arr[i + (j * x)] - mean;
    /* Calculate standard deviation */
    sig *= sig;
    sig /= WINDOW;
    avg[i] = mean;
    dev[i] = SQRT_FUNC(sig);
  } else if (i < xy) {
    /* Avoid NULL reference */
    avg[i] = 0;
    dev[i] = 0;
  }
}

__global__
/**
 * minmax
 * ------
 * Calculates the minimum and maximum values for each feature of the
 * input dataset.
 *
 * @param min array of output values representing minimum value of sliding
 * window (row major order)
 * @param max array of output values representign maximum value of sliding
 * window (column major order)
 * @param arr input array, row major order
 * @param x   width of data
 * @param y   length of data
 */
void minmax(
  DATA_T *min, DATA_T *max, const DATA_T *arr,
  int x, int y, int xy) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    /* Filter over-allocated threads, but ignore last (window - 1) records */
    if (i + (WINDOW * x) < xy) {
      int j;
      DATA_T val, mn, mx;
      mn = mx = val = arr[i + (j * x)];
      for (j = 1; j <= WINDOW; j++) {
        if (mn > val) mn = val;
        if (mx < val) mx = val;
        val = arr[i + (j * x)];
      }
      min[i] = mn;
      max[i] = mx;
    }
  }
