#ifndef FUSION_MACRO_H
#define FUSION_MACRO_H

#ifdef __CUDACC__
#define FUSION_HOST __host__
#define FUSION_DEVICE __device__
#define FUSION_HOST_AND_DEVICE __host__ __device__
#else
#define FUSION_HOST
#define FUSION_DEVICE
#define FUSION_HOST_AND_DEVICE
#endif

#define WARP_SIZE 32
#define MAX_THREAD 1024

#define FUSION_EXPORT

#include <iostream>

#endif