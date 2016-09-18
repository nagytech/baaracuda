#ifndef _RUNNER_H_
#define _RUNNER_H_

#include "const.h"

#define DTH                           cudaMemcpyDeviceToHost
#define HTD                           cudaMemcpyHostToDevice

#define ERR_CALC_FAIL_M               "Device kernel failed: %s\n"
#define ERR_OOM_DEVICE_M              "Device out of memory: %s\n"
#define ERR_OOM_HOST_M                "Host out of memory: %s\n"
#define ERR_MEMCPY_FAILED             "Failed to copy from host to device: %s\n"
#define FUNC_T_AMI                    "averageMovementIntensity"
#define FUNC_T_AVG                    "standardDeviation (average)"
#define FUNC_T_DEV                    "standardDeviation"
#define FUNC_T_MAG                    "signalMagnitude"

int do_ami(DATA_T *data, DATA_T **ami, int x, int y);
int do_calcs(DATA_T *data, DATA_T **mag, DATA_T **ami, DATA_T **dev,
  DATA_T **avg, int x, int y);
int do_dev(DATA_T *data, DATA_T **dev, DATA_T **avg, int x, int y);
int do_mag(DATA_T *data, DATA_T **mag, int x, int y);

#endif
