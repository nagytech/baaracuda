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
 * CUDA error messages.
 *
 * ------------------------------------------------------------------------ */

int do_mag(DATA_T *data, DATA_T *mag, int x, int y) {

  int e;
  DATA_T *d_mag = NULL;

  e = cudaMalloc((void **)&d_mag, y * ct_size);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_MAG);
    return e;
  }

  e = signalMagnitude<<<bpg_multi, TPB>>>(d_mag, d_arr, x, y);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_CALC_FAIL_M, FUNC_T_MAG);
    cudaFree(d_mag);
    return e;
  }

  mag = (DATA_T *)calloc(y, ct_size);
  if (mag == NULL) {
    fprintf(stderr, ERR_OOM_HOST_M, FUNC_T_MAG);
    cudaFree(d_mag);
    return NULL;
  }

  e = cudaMemcpy(mag, d_mag, y * ct_size, dth);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_MEMCPY_FAILED, FUNC_T_MAG);
    free(mag);
    cudaFree(d_mag);
    return e;
  }

  cudaFree(d_mag);

  return e;
}

int do_ami(DATA_T *data, DATA_T *ami, int x, int y) {

  int e;
  DATA_T *d_ami = NULL;

  e = cudaMalloc((void **)&d_ami, y * ct_size);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_AMI);
    return e;
  }

  e = averageMovementIntensity<<<bpg_multi, TPB>>>(d_ami, d_arr, x, y);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_CALC_FAIL_M, FUNC_T_AMI);
    cudaFree(d_ami);
    return e;
  }

  ami = (DATA_T *)calloc(y, ct_size);
  if (ami == NULL) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_AMI);
    cudaFree(d_ami);
    return NULL;
  }

  e = cudaMemcpy(ami, d_ami, y * ct_size, dth);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_MEMCPY_FAILED, FUNC_T_AMI);
    free(ami);
    cudaFree(d_ami);
    return e;
  }

  cudaFree(d_ami);

  return e;

}

int do_dev(DATA_T *data, DATA_T *dev, DATA_T *avg, int x, int y) {

  int e;
  DATA_T *d_dev = NULL, *d_avg = NULL, *dev = NULL, *avg = NULL;

  e = cudaMalloc((void **)&d_dev, x * y * ct_size);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_DEV);
    return e;
  }

  e = cudaMalloc((void **)&d_avg, x * y * ct_size);
  if (e != cudaSuccess) {
    cudaFree(d_dev);
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_AVG);
    return e;
  }

  standardDeviation<<<bpg_singl, TPB>>>(d_dev, d_avg, d_arr, x, y, x * y);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_CALC_FAIL_M, FUNC_T_DEV);
    cudaFree(d_dev);
    cudaFree(d_avg);
    return e;
  }

  dev = (DATA_T *)calloc(x * y, ct_size);
  if (dev == NULL) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_DEV);
    cudaFree(d_dev);
    cudaFree(d_avg);
    return NULL;
  }

  avg = (DATA_T *)calloc(x * y, ct_size);
  if (avg == NULL) {
    fprintf(stderr, ERR_OOM_DEVICE_M, FUNC_T_DEV);
    free(dev);
    cudaFree(d_dev);
    cudaFree(d_avg);
    return NULL;
  }

  e = cudaMemcpy(dev, d_dev, x * y * ct_size, dth);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_CALC_FAIL_M, FUNC_T_DEV);
    free(dev);
    free(avg);
    cudaFree(d_dev);
    cudaFree(d_avg);
    return e;
  }

  e = cudaMemcpy(avg, d_avg, x * y * ct_size, dth);
  if (e != cudaSuccess) {
    fprintf(stderr, ERR_CALC_FAIL_M, FUNC_T_DEV);
    free(dev);
    free(avg);
    cudaFree(d_dev);
    cudaFree(d_avg);
    return e;
  }

  cudaFree(d_dev);
  cudaFree(d_avg);

  return e;

}

int do_calcs(DATA_T *data, DATA_T *mag, DATA_T *ami, DATA_T *dev, DATA_T *avg) {

  size_t ct_size;
  int bpg_multi, bpg_singl;
  cudaMemcpyKind dth, htd;
  DATA_T *d_arr;

  ct_size = sizeof(DATA_T);

  /* Set CUDA Thread / Block Limits */
  bpg_multi = (y + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  bpg_singl = ((x * y) + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  /* Shorten references */
  htd = cudaMemcpyHostToDevice;
  dth = cudaMemcpyDeviceToHost;

  /* Copy the input array over to the device */
  *d_arr = NULL;
  if (cudaMalloc((void **)&d_arr, x * y * ct_size) != cudaSuccess) {
    return EXIT_FAILURE;
  }
  if (cudaMemcpy(d_arr, arr, x * y * ct_size, htd) != cudaSuccess) {
    return EXIT_FAILURE;
  }

  /* Perform the signalMagnitude calculation */
  if (do_mag(data, mag, x, y) != cudaSuccess) {
    cudaFree(d_arr);
    return EXIT_FAILURE;
  }

  /* Perform the averageMovementIntensity calculation */
  if (do_ami(data, ami, x, y) != cudaSuccess) {
    cudaFree(d_arr);
    return EXIT_FAILURE;
  }

  /* Perform the standardDeviation / mean calculation */
  if (do_dev(data, dev, avg, x, y) != cudaSuccess) {
    cudaFree(d_arr);
    return EXIT_FAILURE;
  }

  /* Finalize usage of the data array */
  cudaFree(d_arr);

  return EXIT_SUCCESS;

}
