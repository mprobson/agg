// Description
// TODO
// - define to typedef?
// - Timings
// - Correctness

#define precision_t float

__global__ 
void outer(precision_t* m, precision_t* n, precision_t* mn) {
}

int main(int argc, char* argv[]) {
  // Defaults
  // - k kernels
  int k = 1;
  // - m data on GPU
  size_t m = 1000;
  precision_t* d_m;
  // - n data on host
  size_t n = 1000;
  precision_t* h_n;
  precision_t* d_n;
  // - mn final matrix
  precision_t d_mn;
  precision_t h_mn;

  // Process Input
  if (argc > 1) {
    // TODO
  }
  // Allocate Memory
  // - Device
  cudaMalloc(&d_m, m * sizeof(precision_t));
  // - Host
#if 0
  cudaMallocManaged();
  malloc();
#else
  cudaMallocHost(&h_n, n * sizeof(precision_t));
  cudaMalloc(&d_n, n * sizeof(precision_t));
#endif // 1
  // - Matrix
  cudaMallocHost(&h_mn, m * n * sizeof(precision_t));
  cudaMalloc(&d_mn, m * n * sizeof(precision_t));

  // Generate Two Vectors
  // - Device
  // <<<kernel launch>>>
  // - Host
  for (int i = 0; i < n; i++) {
    //h_n[i] = rand;
  }

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
  // mutiply by scalalr and print? Do some norm/reduction?
}
