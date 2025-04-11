#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <string.h>
/* --- GLOBALS --- */
#define inputSize 784
#define hiddenUnits 28
#define batchSize 32
typedef struct {
  double pixels[784];
  int label;
} ImageVector;
typedef struct {
  double* hiddenActivations;
  double outputActivations;
} ActivationsOutput;

// Helper functions for sigmoid and sigmoid derivative
__device__ __host__ double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-x));
}
__device__ __host__ double sigmoid_derivative(double x) {
    double s = 1.0 / (1.0 + exp(-x));
    return s * (1.0 - s);
}

/* 
  This is a just a reference function for developing the CUDA kernels.
 */
ActivationsOutput* compActivationsCPU(ImageVector* images, int numImages, double** weights1, double* bias1, double* weights2, double bias2){
  ActivationsOutput* output = (ActivationsOutput*)malloc(numImages * sizeof(ActivationsOutput));
  
  for(int i = 0; i < numImages; i++){
    output[i].hiddenActivations = (double*)malloc(hiddenUnits * sizeof(double));
    for(int j = 0; j < hiddenUnits; j++){
      double temp = 0.0;
      for(int k = 0; k < inputSize; k++){
        temp += weights1[k][j] * images[i].pixels[k];
      }
      output[i].hiddenActivations[j] = 1.0 / (1.0 + exp(-(temp + bias1[j])));
    }
    double temp = 0.0;
    for(int j = 0; j < hiddenUnits; j++){
      temp += weights2[j] * output[i].hiddenActivations[j];
    }
    output[i].outputActivations = 1.0 / (1.0 + exp(-(temp + bias2)));
  }
  return output;
}

/*
  This is a wrapper kernel that calls both computation kernels. We can use shared memory here for better performance,
  but for simplicity, and because the performance difference is negligible for MNIST, I'm breaking this into two separate
  kernels that will be called from host.
*/
__global__ void forwardGPU(ImageVector* d_images, double* d_weights1_flat, double* d_bias1, double* d_weights2, 
  double d_bias2, double* d_hiddenActivations, double* d_outputActivations, int startIdx, int numImages) {
  int batchIdx = blockIdx.x;
  int threadId = threadIdx.x;
  
  // First compute hidden activations
  if (threadId < hiddenUnits && batchIdx < batchSize && (startIdx + batchIdx) < numImages) {
    double sum = 0.0;
    for (int i = 0; i < inputSize; i++) {
      sum += d_weights1_flat[i * hiddenUnits + threadId] * d_images[batchIdx].pixels[i];
    }
    d_hiddenActivations[batchIdx * hiddenUnits + threadId] = sigmoid(sum + d_bias1[threadId]);
  }
  
  // Synchronize threads before computing output
  __syncthreads();
  
  // Only one thread per image needs to compute the output
  if (threadId == 0 && batchIdx < batchSize && (startIdx + batchIdx) < numImages) {
    double sum = 0.0;
    for (int i = 0; i < hiddenUnits; i++) {
      sum += d_weights2[i] * d_hiddenActivations[batchIdx * hiddenUnits + i];
    }
    d_outputActivations[batchIdx] = sigmoid(sum + d_bias2);
  }
}

__global__ void updateWeightsGPU(ImageVector* d_images, double* d_weights1_flat, double* d_bias1, double* d_weights2, double* d_bias2,
  double* d_hiddenActivations, double* d_outputActivations, double learning_rate, int startIdx, int numImages) {
  int batchIdx = blockIdx.x;
  int threadId = threadIdx.x;

  // Ensure we stay within batch parameters
  if (batchIdx >= batchSize || (startIdx + batchIdx) >= numImages) return;

  int currLabel = d_images[batchIdx].label;
  double output = d_outputActivations[batchIdx];
  double output_error = output - (double)currLabel; 
  double output_delta = output_error * output * (1.0 - output);

  if (threadId < hiddenUnits) {
    // Update weights2 (28)
    double hidden_activation = d_hiddenActivations[batchIdx * hiddenUnits + threadId];
    atomicAdd(&d_weights2[threadId], -learning_rate * output_delta * hidden_activation);

    double hidden_delta = output_delta * d_weights2[threadId] * hidden_activation * (1.0 - hidden_activation);

    // Update bias1 (28)
    atomicAdd(&d_bias1[threadId], -learning_rate * hidden_delta);

    // Update weights1 (784 * 28)
    for (int i = 0; i < inputSize; i++) {
      atomicAdd(&d_weights1_flat[i * hiddenUnits + threadId], 
        -learning_rate * hidden_delta * d_images[batchIdx].pixels[i]);
    }
  }

  // Finally update bias2 on single thread
  if (threadId == 0) {
    atomicAdd(d_bias2, -learning_rate * output_delta);
  }
}

// Unified function to perform forward pass and weight updates; Returns loss
extern "C" double trainNetworkBatchGPU(ImageVector* images, int numImages, 
  double** weights1, double* bias1, 
  double* weights2, double* bias2,
  double learning_rate) {
  // flatten weights1 for easier GPU memory management
  double* weights1_flat = (double*)malloc(inputSize * hiddenUnits * sizeof(double));
  for (int i = 0; i < inputSize; i++) {
    for (int j = 0; j < hiddenUnits; j++) {
     weights1_flat[i * hiddenUnits + j] = weights1[i][j];
    }
  }

  // gpu memory pointers
  ImageVector* d_images;
  double* d_weights1_flat;
  double* d_bias1;
  double* d_weights2;
  double* d_bias2;
  double* d_hiddenActivations;
  double* d_outputActivations;

  cudaError_t err;
  err = cudaMalloc((void**)&d_weights1_flat, inputSize * hiddenUnits * sizeof(double));
  if (err != cudaSuccess) { printf("cudaMalloc failed: %s\n", cudaGetErrorString(err)); exit(1); }

  err = cudaMalloc((void**)&d_bias1, hiddenUnits * sizeof(double));
  if (err != cudaSuccess) { printf("cudaMalloc failed: %s\n", cudaGetErrorString(err)); exit(1); }

  err = cudaMalloc((void**)&d_weights2, hiddenUnits * sizeof(double));
  if (err != cudaSuccess) { printf("cudaMalloc failed: %s\n", cudaGetErrorString(err)); exit(1); }

  err = cudaMalloc((void**)&d_bias2, sizeof(double));
  if (err != cudaSuccess) { printf("cudaMalloc failed: %s\n", cudaGetErrorString(err)); exit(1); }

  err = cudaMalloc((void**)&d_outputActivations, numImages * sizeof(double));
  if (err != cudaSuccess) { printf("cudaMalloc failed: %s\n", cudaGetErrorString(err)); exit(1); }

  cudaMemcpy(d_weights1_flat, weights1_flat, inputSize * hiddenUnits * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias1, bias1, hiddenUnits * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_weights2, weights2, hiddenUnits * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias2, bias2, sizeof(double), cudaMemcpyHostToDevice);

  dim3 blockDim(batchSize);
  
  // batch processing
  for (int i = 0; i < numImages; i += batchSize) {
    int currentBatchSize = (i + batchSize <= numImages) ? batchSize : (numImages - i);
    dim3 gridDim(currentBatchSize); 
    
    err = cudaMalloc((void**)&d_images, currentBatchSize * sizeof(ImageVector));
    if (err != cudaSuccess) { printf("cudaMalloc failed: %s\n", cudaGetErrorString(err)); exit(1); }
    
    cudaMemcpy(d_images, &images[i], currentBatchSize * sizeof(ImageVector), cudaMemcpyHostToDevice);
    
    err = cudaMalloc((void**)&d_hiddenActivations, currentBatchSize * hiddenUnits * sizeof(double));
    if (err != cudaSuccess) { printf("cudaMalloc failed: %s\n", cudaGetErrorString(err)); exit(1); }

    forwardGPU<<<gridDim, blockDim>>>(
      d_images, 
      d_weights1_flat, 
      d_bias1,
      d_weights2, 
      *bias2, 
      d_hiddenActivations, 
      &d_outputActivations[i],
      i, 
      numImages
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Forward pass kernel error: %s\n", cudaGetErrorString(err));
      exit(1);
    }
    
    cudaDeviceSynchronize(); // Before going to update weights

    updateWeightsGPU<<<gridDim, blockDim>>>(
      d_images, 
      d_weights1_flat, 
      d_bias1,
      d_weights2, 
      d_bias2, 
      d_hiddenActivations, 
      &d_outputActivations[i],
      learning_rate, 
      i, 
      numImages
    );

    err = cudaGetLastError();
    if (err != cudaSuccess) {
      printf("Weight update kernel error: %s\n", cudaGetErrorString(err));
      exit(1);
    }
    
    cudaFree(d_images);
    cudaFree(d_hiddenActivations);
  }

  // copy back updated weights and biases
  cudaMemcpy(weights1_flat, d_weights1_flat, inputSize * hiddenUnits * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(bias1, d_bias1, hiddenUnits * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(weights2, d_weights2, hiddenUnits * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(bias2, d_bias2, sizeof(double), cudaMemcpyDeviceToHost);

  double* activationsToOut = (double*)malloc(numImages * sizeof(double));
  cudaMemcpy(activationsToOut, d_outputActivations, numImages * sizeof(double), cudaMemcpyDeviceToHost);

  double loss = 0.0;
  for (int i = 0; i < numImages; i++) {
    double diff = images[i].label - activationsToOut[i];
    loss += 0.5 * diff * diff;
  }

  // restore 2D weights1 from flat
  for (int i = 0; i < inputSize; i++) {
    for (int j = 0; j < hiddenUnits; j++) {
      weights1[i][j] = weights1_flat[i * hiddenUnits + j];
    }
  }

  cudaFree(d_weights1_flat);
  cudaFree(d_bias1);
  cudaFree(d_weights2);
  cudaFree(d_bias2);
  cudaFree(d_outputActivations);
  free(weights1_flat);
  free(activationsToOut);

  return loss;
}