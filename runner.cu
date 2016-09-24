/**
 * runner.cpp
 * ----------
 *
 * Author: Jonathan Nagy <jnagy@myune.edu.au>
 * Date:   18 Sep 2016
 * Description:
 *
 *    Orchestrates the execution and error checking of the CUDA kernels
 *
 * TODO: More in depth error checking, get the actual error codes and output
 *       CUDA error messages.
 *
 * ------------------------------------------------------------------------ */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "const.h"
#include "kernels.h"
#include "runner.h"

/**
 * do_mag
 * ------
 * Performs the signalMagnitude calculations
 *
 * @param  data input dataset
 * @param  mag  output
 * @param  x    width of dataset (columns)
 * @param  y    length of dataset (rows)
 *
 * @return      cudaSuccess or other
 */
int do_mag(DATA_T *d_arr, DATA_T **mag, int x, int y) {

  int e, bpg_multi;
  DATA_T *d_mag = NULL;

  /* Set block limit */
  bpg_multi = (y + TPB - 1) / TPB;

  /* Allocate device memory */
  e = cudaMalloc((void **)&d_mag, y * sizeof(DATA_T));
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_MAG);
    return e;
  }

  /* Perform calculations */
  signalMagnitude<<<bpg_multi, TPB>>>(d_mag, d_arr, x, y);

  /* Allocate host data */
  *mag = (DATA_T *)calloc(y, sizeof(DATA_T));
  if (*mag == NULL) {
    fprintf(stderr, ERR_OOM_HOST_M, FUNC_T_MAG);
    cudaFree(d_mag);
    return -1;
  }

  /* Copy device to host */
  e = cudaMemcpy(*mag, d_mag, y * sizeof(DATA_T), DTH);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_MEMCPY_FAILED, FUNC_T_MAG);
    free(*mag);
    cudaFree(d_mag);
    return e;
  }

  cudaFree(d_mag);

  return e;
}

/**
 * do_ami
 * ------
 * Performs the averageMovementIntensity calculations
 *
 * @param  data input dataset
 * @param  ami  output
 * @param  x    width of dataset (columns)
 * @param  y    length of dataset (rows)
 *
 * @return      cudaSuccess or other
 */
int do_ami(DATA_T *d_arr, DATA_T **ami, int x, int y) {

  int e, bpg_multi;
  DATA_T *d_ami = NULL;

  /* Set block limit */
  bpg_multi = (y + TPB - 1) / TPB;

  /* Allocate device memory */
  e = cudaMalloc((void **)&d_ami, y * sizeof(DATA_T));
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_AMI);
    return e;
  }

  /* Perform calculations */
  averageMovementIntensity<<<bpg_multi, TPB>>>(d_ami, d_arr, x, y);

  /* Allocate host memory */
  *ami = (DATA_T *)calloc(y, sizeof(DATA_T));
  if (*ami == NULL) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_AMI);
    cudaFree(d_ami);
    return -1;
  }

  /* Copy device to host */
  e = cudaMemcpy(*ami, d_ami, y * sizeof(DATA_T), DTH);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_MEMCPY_FAILED, FUNC_T_AMI);
    free(*ami);
    cudaFree(d_ami);
    return e;
  }

  cudaFree(d_ami);

  return e;

}

/**
 * do_dev
 * ------
 * Performs the standardDeviation calculations (including mean)
 *
 * @param  data input dataset
 * @param  dev  output for standardDeviation
 * @param  avg  output for average
 * @param  x    width of dataset (columns)
 * @param  y    length of dataset (rows)
 *
 * @return      cudaSuccess or other
 */
int do_dev(DATA_T *d_arr, DATA_T **dev, DATA_T **avg, int x, int y) {

  int e, bpg_singl;
  DATA_T *d_dev = NULL, *d_avg = NULL;

  /* Set block limit */
  bpg_singl = ((x * y) + TPB - 1) / TPB;

  /* Allocate device memory */
  e = cudaMalloc((void **)&d_dev, x * y * sizeof(DATA_T));
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_DEV);
    return e;
  }
  e = cudaMalloc((void **)&d_avg, x * y * sizeof(DATA_T));
  if (e != cudaSuccess) {
    cudaFree(d_dev);
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_AVG);
    return e;
  }

  /* Perform calculations */
  standardDeviation<<<bpg_singl, TPB>>>(d_dev, d_avg, d_arr, x, y, x * y);

  /* Allocate host memory */
  *dev = (DATA_T *)calloc(x * y, sizeof(DATA_T));
  if (*dev == NULL) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_DEV);
    cudaFree(d_dev);
    cudaFree(d_avg);
    return -1;
  }
  *avg = (DATA_T *)calloc(x * y, sizeof(DATA_T));
  if (*avg == NULL) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_DEV);
    free(*dev);
    cudaFree(d_dev);
    cudaFree(d_avg);
    return -1;
  }

  /* Copy host to device */
  e = cudaMemcpy(*dev, d_dev, x * y * sizeof(DATA_T), DTH);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_CALC_FAIL_M, FUNC_T_DEV);
    free(*dev);
    free(*avg);
    cudaFree(d_dev);
    cudaFree(d_avg);
    return e;
  }
  e = cudaMemcpy(*avg, d_avg, x * y * sizeof(DATA_T), DTH);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_CALC_FAIL_M, FUNC_T_DEV);
    free(*dev);
    free(*avg);
    cudaFree(d_dev);
    cudaFree(d_avg);
    return e;
  }

  cudaFree(d_dev);
  cudaFree(d_avg);

  return e;

}

/**
 * do_minmax
 * ---------
 * Calculates the min and max of the sliding window for the input dataset
 * features.
 *
 * @param  d_arr input data array
 * @param  min   output minimum data array
 * @param  max   output maximum data array
 * @param  x     dataset width
 * @param  y     dataset length
 *
 * @return       success or failure
 */
int do_minmax(DATA_T *d_arr, DATA_T **min, DATA_T **max, int x, int y) {

  int e, bpg_singl;
  DATA_T *d_min = NULL, *d_max = NULL;

  /* Set block limit */
  bpg_singl = ((x * y) + TPB - 1) / TPB;

  /* Allocate device memory */
  e = cudaMalloc((void **)&d_min, x * y * sizeof(DATA_T));
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_MIN);
    return e;
  }
  e = cudaMalloc((void **)&d_max, x * y * sizeof(DATA_T));
  if (e != cudaSuccess) {
    cudaFree(d_min);
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_MAX);
    return e;
  }

  /* Perform calculations */
  minmax<<<bpg_singl, TPB>>>(d_min, d_max, d_arr, x, y, x * y);

  /* Allocate host memory */
  *min = (DATA_T *)calloc(x * y, sizeof(DATA_T));
  if (*min == NULL) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_MIN);
    cudaFree(d_min);
    cudaFree(d_max);
    return -1;
  }
  *max = (DATA_T *)calloc(x * y, sizeof(DATA_T));
  if (*max == NULL) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_MAX);
    free(*min);
    cudaFree(d_min);
    cudaFree(d_max);
    return -1;
  }

  /* Copy host to device */
  e = cudaMemcpy(*min, d_min, x * y * sizeof(DATA_T), DTH);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_CALC_FAIL_M, FUNC_T_MIN);
    free(*min);
    free(*max);
    cudaFree(d_min);
    cudaFree(d_max);
    return e;
  }
  e = cudaMemcpy(*max, d_max, x * y * sizeof(DATA_T), DTH);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_CALC_FAIL_M, FUNC_T_MAX);
    free(*min);
    free(*max);
    cudaFree(d_min);
    cudaFree(d_max);
    return e;
  }

  cudaFree(d_min);
  cudaFree(d_max);

  return e;

}

/**
 * do_calcs
 * --------
 *
 * Orchestrates the CUDA kernel execution for all calculations
 *
 * @param  data input data
 * @param  mag  magnitude output
 * @param  ami  averageMovementIntensity output
 * @param  dev  standardDeviation output
 * @param  avg  standardDeviation output (mean)
 *
 * @return      success or failure
 */
int do_calcs(DATA_T *data, DATA_T **mag, DATA_T **ami, DATA_T **dev,
  DATA_T **avg, DATA_T **min, DATA_T **max, int x, int y) {

  int e;
  DATA_T *d_arr = NULL;

  /* Copy the input array over to the device */
  if (cudaMalloc((void **)&d_arr, x*y * sizeof(DATA_T)) != cudaSuccess) {
    return EXIT_FAILURE;
  }

  e = cudaMemcpy(d_arr, data, x * y * sizeof(DATA_T), cudaMemcpyHostToDevice);
  if (e != cudaSuccess) {
    return EXIT_FAILURE;
  }

  /* Perform the signalMagnitude calculation */
  if (do_mag(d_arr, mag, x, y) != cudaSuccess) {
    cudaFree(d_arr);
    return EXIT_FAILURE;
  }

  /* Perform the averageMovementIntensity calculation */
  if (do_ami(d_arr, ami, x, y) != cudaSuccess) {
    cudaFree(d_arr);
    return EXIT_FAILURE;
  }

  /* Perform the standardDeviation / mean calculation */
  if (do_dev(d_arr, dev, avg, x, y) != cudaSuccess) {
    cudaFree(d_arr);
    return EXIT_FAILURE;
  }

  /* Perform the min max calculations */
  if (do_minmax(d_arr, min, max, x, y) != cudaSuccess) {
    cudaFree(d_arr);
    return EXIT_FAILURE;
  }

  /* Finalize usage of the data array */
  cudaFree(d_arr);

  return EXIT_SUCCESS;

}
