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
#include <cstdio>
#include <ctime>
#include <unistd.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define PRINT_STREAM 1

typedef float precision_t;

double hostElapsedTimeMs(timespec start, timespec stop) {
  return (stop.tv_sec  - start.tv_sec)  * 1000 +
         (stop.tv_nsec - start.tv_nsec) / 1000000.;
}

__device__
unsigned mySmId() {
  unsigned smId;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smId));
  return smId;
}

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
    size_t m, size_t n,
#if PRINT_STREAM
    int streamId
#endif
    ) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Print SM
#if PRINT_STREAM
  if(threadIdx.x == 0) printf("Kernel: %d\tBlock: %d\tSM %d\n", streamId, blockIdx.x, mySmId());
#else
  if(threadIdx.x == 0) printf("SM %d\n", mySmId());
#endif

  // TODO transform loop for coalesced writes? and/or use shared mem?
  for (int i = idx; i < m; i += gridDim.x * blockDim.x) {
    for (int j = 0; j < n; j ++) {
      d_mn[i * m + j] = d_m[i] * d_n[j];
    }
  }
}

int main(int argc, char* argv[]) {
  // Defaults
  // TODO unsigned for all ints?
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

  // - number of warmup iterations
  int nWarmupIter = 1000;

  unsigned int hSeed = std::time(NULL);
  unsigned int dSeed = hSeed;

  // Process Input
  // TODO add static cast to atoi?
  int c;
  if (argc > 1) {
    while((c = getopt(argc, argv, "k:m:n:t:r:w:h:d:")) != -1) {
      switch (c) {
        case 'k':
          numKernels = atoi(optarg);
          break;
        case 'm':
          m = atoi(optarg);
          break;
        case 'n':
          n = atoi(optarg);
          break;
        case 't':
          threadsPerBlock = atoi(optarg);
          break;
        case 'r':
          rMax = atoi(optarg);
          break;
        case 'w':
          nWarmupIter = atoi(optarg);
          break;
        case 'h':
          hSeed = atoi(optarg);
          break;
        case 'd':
          dSeed = atoi(optarg);
          break;
        case '?':
          printf("Ignoring unrecognized option: '-%c'\n", optopt);
          break;
        default:
          printf("Problem parsing input\n");
          break;
      }
    }
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

  // Allocate Streams
  cudaStream_t streams[numKernels];
  for (int i = 0; i < numKernels; i++) {
    cudaStreamCreate(&streams[i]);
  }

  // TODO implement rounding for num blocks launched
  init<<<m/threadsPerBlock, threadsPerBlock>>>(dSeed, d_states);

  // Generate Two Vectors
  // - Device
  // Warmup
  //for (int i = 0; i < nWarmupIter; i++) {
    generate<<<m/threadsPerBlock, threadsPerBlock>>>(d_m, m, rMax, d_states);
  //}

  // - Host
  std::srand(hSeed);
  for (int i = 0; i < n; i++) {
    h_n[i] = std::rand() / ((RAND_MAX + 1u) / rMax);
  }

  // Ensure random number generation finishes
  cudaDeviceSynchronize();

  // Timing
  timespec hStart, hStop;
  double hTimeMs [3];

  cudaEvent_t dStart, dStop;
  cudaEventCreate(&dStart);
  cudaEventCreate(&dStop);
  float dTimeMs [3];

  // Copy Warmup
  for(int i = 0; i < nWarmupIter; i++) {
#if 0
    cudaMemcpyAsync();
#else
    cudaMemcpy(d_n, h_n, sizeof(precision_t), cudaMemcpyHostToDevice);
#endif
  }

  // - events
  cudaEventRecord(dStart);

  // - host side
  // TODO check return value for errors
  clock_gettime(CLOCK_MONOTONIC, &hStart);

  // Copy Vector
#if 0
  cudaMemcpyAsync();
#else
  cudaMemcpy(d_n, h_n, n * sizeof(precision_t), cudaMemcpyHostToDevice);
#endif // 1
  // Timing
  // - host side
  clock_gettime(CLOCK_MONOTONIC, &hStop);
  // - events
  cudaEventRecord(dStop);

  cudaEventSynchronize(dStop);

  // Elapsed Time
  cudaEventElapsedTime(&(dTimeMs[0]), dStart, dStop);
  hTimeMs[0] = hostElapsedTimeMs(hStart, hStop);

  // Warmup
  for (int i = 0; i < nWarmupIter; i++) {
    empty<<<1,1>>>();
  }

  cudaEventRecord(dStart);
  clock_gettime(CLOCK_MONOTONIC, &hStart);

  // Execute
  for (int i = 0; i < numKernels; i++) {
    // TODO add offsets into matricies for multiple kernels
#if PRINT_STREAM
    outer<<<m/threadsPerBlock, threadsPerBlock, 0, streams[i]>>>(d_m, d_n, d_mn, m, n, i);
#else
    outer<<<m/threadsPerBlock, threadsPerBlock, 0, streams[i]>>>(d_m, d_n, d_mn, m, n);
#endif
  }

  cudaDeviceSynchronize();

  clock_gettime(CLOCK_MONOTONIC, &hStop);
  cudaEventRecord(dStop);

  cudaEventSynchronize(dStop);

  cudaEventElapsedTime(&(dTimeMs[1]), dStart, dStop);
  hTimeMs[1] = hostElapsedTimeMs(hStart, hStop);

  cudaEventRecord(dStart);
  clock_gettime(CLOCK_MONOTONIC, &hStart);

  // Copy Back
#if 0
  cudaMemcpyAsync();
#else
  cudaMemcpy(h_mn, d_mn, m * n * sizeof(precision_t), cudaMemcpyDeviceToHost);
#endif // 1

  clock_gettime(CLOCK_MONOTONIC, &hStop);
  cudaEventRecord(dStop);

  cudaEventSynchronize(dStop);

  cudaEventElapsedTime(&(dTimeMs[2]), dStart, dStop);
  hTimeMs[2] = hostElapsedTimeMs(hStart, hStop);

  printf("Parameters\n"
         "\t numKernels:      %d\n"
         "\t m:               %zu\n"
         "\t n:               %zu\n"
         "\t threadsPerBlock: %zu\n"
         "\t randomMax:       %d\n"
         "\t nWarmupIter:     %d\n"
         "\t hostSeed:        %u\n"
         "\t deviceSeed:      %u\n",
         numKernels, m, n, threadsPerBlock, rMax, nWarmupIter, hSeed, dSeed);

  printf("Timing   \tHost\t\tDevice\n"
         "Copy In: \t%f  \t%f\n"
         "Execute: \t%f  \t%f\n"
         "Copy Out:\t%f  \t%f\n",
          hTimeMs[0], dTimeMs[0],
          hTimeMs[1], dTimeMs[1],
          hTimeMs[2], dTimeMs[2]);

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
