#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_DIM 8
#define KERNEL_DIM 3
#define RADIUS 1

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[KERNEL_DIM][KERNEL_DIM][KERNEL_DIM];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
  int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
  int x_o = bx * TILE_DIM + tx;
  int y_o = by * TILE_DIM + ty;
  int z_o = bz * TILE_DIM + tz;
  int x_i = x_o - RADIUS;
  int y_i = y_o - RADIUS;
  int z_i = z_o - RADIUS;

  __shared__ float input_tile[TILE_DIM + 2 * RADIUS][TILE_DIM + 2 * RADIUS][TILE_DIM + 2 * RADIUS];
  if (x_i >= 0 && x_i < x_size && y_i >= 0 && y_i < y_size && z_i >= 0 && z_i < z_size) {
    input_tile[tz][ty][tx] = input[z_i * (y_size * x_size) + y_i * x_size + x_i];
  } else {
    input_tile[tz][ty][tx] = 0.f;
  }
  __syncthreads();

  if (tz < TILE_DIM && ty < TILE_DIM && tx < TILE_DIM) {
    float pval = 0.f;
    for (int i = 0; i < KERNEL_DIM; i++)
      for (int j = 0; j < KERNEL_DIM; j++)
        for (int k = 0; k < KERNEL_DIM; k++)
          pval += deviceKernel[i][j][k] * input_tile[tz + i][ty + j][tx + k];
    
    if (x_o < x_size && y_o < y_size && z_o < z_size)
      output[z_o * (y_size * x_size) + y_o * x_size + x_o] = pval;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void **) &deviceInput, sizeof(float) * z_size * y_size * x_size);
  cudaMalloc((void **) &deviceOutput, sizeof(float) * z_size * y_size * x_size);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, sizeof(float) * z_size * y_size * x_size, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, sizeof(float) * KERNEL_DIM * KERNEL_DIM * KERNEL_DIM);
  
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(1.0 * x_size / TILE_DIM), ceil(1.0 * y_size / TILE_DIM), ceil(1.0 * z_size / TILE_DIM));
  dim3 dimBlock(TILE_DIM + 2 * RADIUS, TILE_DIM + 2 * RADIUS, TILE_DIM + 2 * RADIUS);

  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, sizeof(float) * z_size * y_size * x_size, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}
