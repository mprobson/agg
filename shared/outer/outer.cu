// Description
//  Outer product
// TODO
// - Kernel Sizes
// - Timings
// Future
// - Correctness
// - (CUDA) Error Checking
// - Checking and Generating Doubles
#include <cstdlib>
#include <ctime>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

typedef float precision_t;

__global__
void empty() {
};

__global__
void init(unsigned int seed, curandState* state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, idx, 0, &state[idx]);
}

__global__
void generate(precision_t* d_m, size_t m, int rMax, curandState* state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curandState localState = state[idx];
  for (int i = idx; i < m ; i += gridDim.x * blockDim.x) {
    d_m[i] = curand_uniform(&localState) * rMax;
  }
  state[idx] = localState;
}

__global__
void outer(precision_t* d_m, precision_t* d_n, precision_t* d_mn,
    size_t m, size_t n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // TODO transform loop for coalesced writes? and/or use shared mem?
  for (int i = idx; i < m; i += gridDim.x * blockDim.x) {
    for (int j = 0; j < n; j ++) {
      d_mn[i * m + j] = d_m[i] * d_n[j];
    }
  }
}

int main(int argc, char* argv[]) {
  // Defaults
  // - number of kernels
  int numKernels = 1;

  // - m data on GPU
  size_t m = 1024;

  // - n data on host
  size_t n = 1024;

  // - threadsPerBlock
  size_t threadsPerBlock = 128;

  // - r max number
  int rMax = 100;

  unsigned int hSeed = std::time(NULL);
  unsigned int dSeed = hSeed;

  // Process Input
  if (argc > 1) {
    // TODO
  }

  // Allocate Memory
#if 0
  cudaMallocManaged();
  malloc();
#else
  // - Device
  precision_t* d_m;
  cudaMalloc(&d_m, m * sizeof(precision_t));

  // - Host
  precision_t* h_n;
  cudaMallocHost(&h_n, n * sizeof(precision_t));
  precision_t* d_n;
  cudaMalloc(&d_n, n * sizeof(precision_t));

  // - Matrix
  precision_t* h_mn;
  cudaMallocHost(&h_mn, m * n * sizeof(precision_t));
  precision_t* d_mn;
  cudaMalloc(&d_mn, m * n * sizeof(precision_t));

  // TODO could clamp state size smaller if we know total num threads
  curandState* d_states;
  cudaMalloc(&d_states, m * sizeof(curandState));
#endif // 1

  // TODO implement rounding for num blocks launched
  init<<<m/threadsPerBlock, threadsPerBlock>>>(dSeed, d_states);

  // Generate Two Vectors
  // - Device
  generate<<<m/threadsPerBlock, threadsPerBlock>>>(d_m, m, rMax, d_states);

  // - Host
  std::srand(hSeed);
  for (int i = 0; i < n; i++) {
    h_n[i] = std::rand() / ((RAND_MAX + 1u) / rMax);
  }

  // Ensure random number generation finishes
  cudaDeviceSynchronize();

  // Copy Warmup
  for(int i = 0; i < 100; i++) {
#if 0
    cudaMemcpyAsync();
#else
    cudaMemcpy(d_n, h_n, sizeof(precision_t), cudaMemcpyHostToDevice);
#endif
  }

  // Copy Vector
#if 0
  cudaMemcpyAsync();
#else
  cudaMemcpy(d_n, h_n, n * sizeof(precision_t), cudaMemcpyHostToDevice);
#endif // 1

  // Warmup
  for (int i = 0; i < 100; i++) {
    empty<<<1,1>>>();
  }

  // Execute
  for (int i = 0; i < numKernels; i++) {
    // TODO add offsets into matricies for multiple kernels
    outer<<<m/threadsPerBlock, threadsPerBlock>>>(d_m, d_n, d_mn, m, n);
  }

  // Copy Back
#if 0
  cudaMemcpyAsync();
#else
  cudaMemcpy(h_mn, d_mn, m * n * sizeof(precision_t), cudaMemcpyDeviceToHost);
#endif // 1

  // Do something with the data to prevent optimization
  // mutiply by scalar and print? Do some norm/reduction?
  precision_t result;
  cublasHandle_t handle;
  cublasStatus_t stat;

  stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf ("CUBLAS initialization failed\n");
    return EXIT_FAILURE;
  }

  cublasSnrm2(handle, m * n, h_mn, 1, &result);
  printf("Result: %f\n", result);

  cublasDestroy(handle);

  cudaFree(d_m);
  cudaFree(d_n);
  cudaFree(d_mn);
  cudaFree(d_states);
  // free/cudaFree h_*

  return EXIT_SUCCESS;
}
