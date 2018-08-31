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

typedef float precision_t;

__global__
void init(unsigned int seed, curandState* state) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, idx, 0, &state[id]);
}

__global__
void generate(precision_t* m, size_t m, int rMax, curandState* state) {
  int idx = blockId.x * blockDim.x + threadIdx.x;
  curandState localState = state[id];
  for (int i = idx; i < m ; i += gridDim.x) {
    m[i] = curand_uniform(&localState) * rMax;
  }
  state[id] = localState;
}

__global__
void outer(precision_t* m, precision_t* n, precision_t* mn) {
}

int main(int argc, char* argv[]) {
  // Defaults
  // - k kernels
  int k = 1;

  // - m data on GPU
  size_t m = 1000;

  // - n data on host
  size_t n = 1000;

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
  precision_t h_mn;
  cudaMallocHost(&h_mn, m * n * sizeof(precision_t));
  precision_t d_mn;
  cudaMalloc(&d_mn, m * n * sizeof(precision_t));

  curandState* d_states;
  cudaMalloc(&d_states, * sizeof(curandState));
#endif // 1

  //init<<<>>>(dSeed, d_states);

  // Generate Two Vectors
  // - Device
  //generate<<<>>>(d_m, m, rMax, d_states);

  // - Host
  std::srand(hSeed);
  for (int i = 0; i < n; i++) {
    h_n[i] = std::rand() / ((RAND_MAX + 1u) / rMax);
  }

  // Ensure random number generation finishes
  cudaDeviceSynchronize();

  // Copy Vector
#if 0
  cudaMemcpyAsync();
#else
  cudaMemcpy(d_n, h_n, n * sizeof(precision_t), cudaMemcpyHostToDevice);
#endif // 1

  // Execute
  //outer<<<>>>(d_m, d_n, d_mn);

  // Copy Back
#if 0
  cudaMemcpyAsync();
#else
  cudaMemcpy(h_mn, d_mn, m * n * sizeof(precision_t), cudaMemcpyDeviceToHost);
#endif // 1

  // Do something with the data to prevent optimization
  // mutiply by scalar and print? Do some norm/reduction?

  cudaFree(d_m);
  cudaFree(d_n);
  cudaFree(d_mn);
  cudaFree(d_states);
  // free/cudaFree h_*
}
