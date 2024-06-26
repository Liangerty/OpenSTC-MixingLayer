#include "kernels.cuh"
#include "Parallel.h"
#include <cstdio>

namespace cfd{
void setup_gpu_device(int n_proc, int myid) {
  int deviceCount{0};
  cudaGetDeviceCount(&deviceCount);

  if (deviceCount < n_proc) {
    printf("Not enough GPU devices.\n"
           "We want %d GPUs but only %d GPUs are available.\n"
           " Stop computing.\n", n_proc, deviceCount);
    MpiParallel::exit();
  }

  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, myid);
  cudaSetDevice(myid);
  printf("\tProcess %d will compute on device [[%s]].\n", myid, prop.name);
}
}
