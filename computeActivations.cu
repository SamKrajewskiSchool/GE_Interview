#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

/* -----------------------------------------------------------------------
 * Mirror the architecture constants from cnn_mnist.c
 * --------------------------------------------------------------------- */
#define INPUT_H       28
#define INPUT_W       28
#define NUM_FILTERS   8
#define FILTER_SIZE   5
#define CONV_OUT_H    (INPUT_H - FILTER_SIZE + 1)                 // 24
#define CONV_OUT_W    (INPUT_W - FILTER_SIZE + 1)                 // 24
#define POOL_SIZE     2
#define POOL_OUT_H    (CONV_OUT_H / POOL_SIZE)                    // 12
#define POOL_OUT_W    (CONV_OUT_W / POOL_SIZE)                    // 12
#define FC_INPUT_SIZE (NUM_FILTERS * POOL_OUT_H * POOL_OUT_W)     // 1152
#define NUM_CLASSES   10
#define BATCH_SIZE    32

/* -----------------------------------------------------------------------
 * Flat array size constants (for cudaMalloc arithmetic)
 * --------------------------------------------------------------------- */
#define FILTERS_SIZE   (NUM_FILTERS * FILTER_SIZE * FILTER_SIZE)  // 200
#define CONV_OUT_SIZE  (NUM_FILTERS * CONV_OUT_H * CONV_OUT_W)    // 4608
#define POOL_OUT_SIZE  (NUM_FILTERS * POOL_OUT_H * POOL_OUT_W)    // 1152
#define POOL_MASK_SIZE (NUM_FILTERS * POOL_OUT_H * POOL_OUT_W)    // 1152  (ints)
#define FC_W_SIZE      (NUM_CLASSES * FC_INPUT_SIZE)              // 11520

/* -----------------------------------------------------------------------
 * Flat CNNParams layout (all doubles, packed for a single cudaMemcpy)
 *
 *   [0             .. FILTERS_SIZE-1          ]  conv filters
 *   [FILTERS_SIZE  .. FILTERS_SIZE+NUM_FILTERS-1] conv biases
 *   [FC_W_OFFSET   .. FC_W_OFFSET+FC_W_SIZE-1 ]  fc weights
 *   [FC_B_OFFSET   .. FC_B_OFFSET+NUM_CLASSES-1]  fc biases
 * --------------------------------------------------------------------- */
#define FC_W_OFFSET  (FILTERS_SIZE + NUM_FILTERS)
#define FC_B_OFFSET  (FC_W_OFFSET + FC_W_SIZE)
#define PARAMS_TOTAL (FC_B_OFFSET + NUM_CLASSES)

typedef struct {
    double pixels[784];
    int    label;
} ImageVector;

/* Host-side CNNParams (must match cnn_mnist.c exactly) */
typedef struct {
    double filters    [NUM_FILTERS][FILTER_SIZE][FILTER_SIZE];
    double conv_bias  [NUM_FILTERS];
    double fc_weights [NUM_CLASSES][FC_INPUT_SIZE];
    double fc_bias    [NUM_CLASSES];
} CNNParams;

/* -----------------------------------------------------------------------
 * Device helpers
 * --------------------------------------------------------------------- */
__device__ __forceinline__ double d_relu(double x)      { return x > 0.0 ? x : 0.0; }
__device__ __forceinline__ double d_relu_grad(double x) { return x > 0.0 ? 1.0 : 0.0; }

/* -----------------------------------------------------------------------
 * Kernel 1: Conv + ReLU + Max-pool (one thread block per image)
 *
 * Grid : (currentBatchSize)
 * Block: (NUM_FILTERS, POOL_OUT_H, POOL_OUT_W)  → 8*12*12 = 1152 threads
 *
 * Each thread is responsible for one (filter, pool_row, pool_col) output cell.
 * It computes the 2x2 max over the corresponding conv outputs, storing both
 * the pooled value and the mask (which of the 4 positions held the max).
 * --------------------------------------------------------------------- */
__global__ void forwardConvPoolGPU(
    const ImageVector* __restrict__ d_images,
    const double*      __restrict__ d_params,         // flat CNNParams
    double*            d_conv_out,                    // [batch, NUM_FILTERS, CONV_OUT_H, CONV_OUT_W]
    double*            d_pool_out,                    // [batch, FC_INPUT_SIZE]
    int*               d_pool_mask,                   // [batch, POOL_MASK_SIZE]
    int numImages, int startIdx)
{
    int batchIdx = blockIdx.x;
    if (batchIdx >= BATCH_SIZE || (startIdx + batchIdx) >= numImages) return;

    /* Thread → (filter f, pool row pi, pool col pj) */
    int f  = threadIdx.x;                             // 0..NUM_FILTERS-1
    int pi = threadIdx.y;                             // 0..POOL_OUT_H-1
    int pj = threadIdx.z;                             // 0..POOL_OUT_W-1

    const double* img = d_images[batchIdx].pixels;

    /* --- Compute the POOL_SIZE x POOL_SIZE conv outputs that feed this pool cell --- */
    int ri = pi * POOL_SIZE;
    int rj = pj * POOL_SIZE;

    double max_val = -1e30;
    int    max_idx = 0;

    for (int di = 0; di < POOL_SIZE; di++) {
        for (int dj = 0; dj < POOL_SIZE; dj++) {
            int ci = ri + di;   // conv output row
            int cj = rj + dj;   // conv output col

            /* Dot product: filter f over patch starting at (ci, cj) */
            double sum = d_params[FILTERS_SIZE + f]; // conv bias
            int filter_base = f * FILTER_SIZE * FILTER_SIZE;
            for (int ki = 0; ki < FILTER_SIZE; ki++) {
                for (int kj = 0; kj < FILTER_SIZE; kj++) {
                    int px = (ci + ki) * INPUT_W + (cj + kj);
                    sum += d_params[filter_base + ki * FILTER_SIZE + kj] * img[px];
                }
            }
            double act = d_relu(sum);

            /* Store conv output for use in backward pass */
            int conv_idx = batchIdx * CONV_OUT_SIZE
                         + f * CONV_OUT_H * CONV_OUT_W
                         + ci * CONV_OUT_W + cj;
            d_conv_out[conv_idx] = act;

            if (act > max_val) { max_val = act; max_idx = di * POOL_SIZE + dj; }
        }
    }

    /* Write pool output and mask */
    int pool_base = batchIdx * POOL_MASK_SIZE
                  + f * POOL_OUT_H * POOL_OUT_W
                  + pi * POOL_OUT_W + pj;
    d_pool_out [pool_base] = max_val;
    d_pool_mask[pool_base] = max_idx;
}

/* -----------------------------------------------------------------------
 * Kernel 2: FC layer + Softmax (one thread block per image)
 *
 * Grid : (currentBatchSize)
 * Block: (NUM_CLASSES)  → 10 threads
 *
 * Each thread computes one class logit, then we do a parallel reduction
 * for the softmax denominator using shared memory.
 * --------------------------------------------------------------------- */
__global__ void forwardFCSoftmaxGPU(
    const double* __restrict__ d_pool_out,   // [batch, FC_INPUT_SIZE]
    const double* __restrict__ d_params,
    double*       d_logits,                  // [batch, NUM_CLASSES]  (pre-softmax)
    double*       d_softmax,                 // [batch, NUM_CLASSES]
    int numImages, int startIdx)
{
    int batchIdx = blockIdx.x;
    if (batchIdx >= BATCH_SIZE || (startIdx + batchIdx) >= numImages) return;

    int c = threadIdx.x;  // class index
    __shared__ double shared_exp[NUM_CLASSES];
    __shared__ double shared_max;

    const double* fc_in = &d_pool_out[batchIdx * FC_INPUT_SIZE];

    /* Compute logit for class c */
    double sum = d_params[FC_B_OFFSET + c]; // fc bias
    int fc_w_base = FC_W_OFFSET + c * FC_INPUT_SIZE;
    for (int i = 0; i < FC_INPUT_SIZE; i++)
        sum += d_params[fc_w_base + i] * fc_in[i];

    d_logits[batchIdx * NUM_CLASSES + c] = sum;
    shared_exp[c] = sum;
    __syncthreads();

    /* Numerically stable softmax: find max */
    if (c == 0) {
        double mx = shared_exp[0];
        for (int i = 1; i < NUM_CLASSES; i++)
            if (shared_exp[i] > mx) mx = shared_exp[i];
        shared_max = mx;
    }
    __syncthreads();

    shared_exp[c] = exp(shared_exp[c] - shared_max);
    __syncthreads();

    /* Sum reduction */
    if (c == 0) {
        double s = 0.0;
        for (int i = 0; i < NUM_CLASSES; i++) s += shared_exp[i];
        shared_exp[0] = s;   // reuse slot 0 for the sum
    }
    __syncthreads();

    d_softmax[batchIdx * NUM_CLASSES + c] = shared_exp[c + (c == 0 ? 0 : 0)] 
                                            / shared_exp[0];
    /* Simpler rewrite (shared_exp[0] now holds total sum): */
    double total = shared_exp[0];
    d_softmax[batchIdx * NUM_CLASSES + c] = exp(d_logits[batchIdx * NUM_CLASSES + c] - shared_max) / total;
}

/* -----------------------------------------------------------------------
 * Kernel 3: Backward pass — FC gradients (one block per image)
 *
 * Grid : (currentBatchSize)
 * Block: (FC_INPUT_SIZE)  — NOTE: FC_INPUT_SIZE=1152 > 1024 (max threads/block)
 *        → launch with block=(NUM_CLASSES) and loop over FC_INPUT_SIZE inside,
 *          or use a 2D block.  We use block=(NUM_CLASSES) + inner loop.
 * --------------------------------------------------------------------- */
__global__ void backwardFCGPU(
    const double* __restrict__ d_pool_out,   // [batch, FC_INPUT_SIZE]
    const double* __restrict__ d_softmax,    // [batch, NUM_CLASSES]
    const int*    __restrict__ d_labels,     // [batch]
    double*       d_params_grad,             // [PARAMS_TOTAL] — accumulate here
    double*       d_d_pool_out,              // [batch, FC_INPUT_SIZE] — gradient to pass back
    int numImages, int startIdx)
{
    int batchIdx = blockIdx.x;
    if (batchIdx >= BATCH_SIZE || (startIdx + batchIdx) >= numImages) return;

    int c = threadIdx.x;  // class index

    int label = d_labels[batchIdx];

    /* dL/d(logit_c) = softmax_c - 1{c==label} */
    double d_logit = d_softmax[batchIdx * NUM_CLASSES + c]
                   - (c == label ? 1.0 : 0.0);

    /* FC bias gradient */
    atomicAdd(&d_params_grad[FC_B_OFFSET + c], d_logit);

    /* FC weight gradients + accumulate d_pool_out gradient */
    int fc_w_base = FC_W_OFFSET + c * FC_INPUT_SIZE;
    const double* fc_in = &d_pool_out[batchIdx * FC_INPUT_SIZE];
    double*       d_in  = &d_d_pool_out[batchIdx * FC_INPUT_SIZE];

    for (int i = 0; i < FC_INPUT_SIZE; i++) {
        atomicAdd(&d_params_grad[fc_w_base + i], d_logit * fc_in[i]);
        atomicAdd(&d_in[i], /* d_params[fc_w_base+i] * d_logit — needs current weights */
                  d_logit); // placeholder: full version below in host wrapper
    }
    /* Note: d_d_pool_out requires the current weight values, so we compute it
       separately in the host wrapper after copying weights to a device buffer.
       For brevity here the accumulation is completed in backwardConvPoolGPU. */
}

/* -----------------------------------------------------------------------
 * Kernel 4: Backward pass — Conv+Pool gradients (one block per image)
 *
 * Grid : (currentBatchSize)
 * Block: (NUM_FILTERS, POOL_OUT_H, POOL_OUT_W)  → 1152 threads
 * --------------------------------------------------------------------- */
__global__ void backwardConvPoolGPU(
    const ImageVector* __restrict__ d_images,
    const double*      __restrict__ d_params,        // current weights (read-only)
    const double*      __restrict__ d_softmax,
    const int*         __restrict__ d_labels,
    const double*      __restrict__ d_conv_out,      // [batch, CONV_OUT_SIZE]
    const int*         __restrict__ d_pool_mask,     // [batch, POOL_MASK_SIZE]
    double*            d_params_grad,                // accumulate gradients
    int numImages, int startIdx)
{
    int batchIdx = blockIdx.x;
    if (batchIdx >= BATCH_SIZE || (startIdx + batchIdx) >= numImages) return;

    int f  = threadIdx.x;
    int pi = threadIdx.y;
    int pj = threadIdx.z;

    int label = d_labels[batchIdx];

    /* --- Recompute d_pool_out gradient for this (f, pi, pj) cell --- */
    /* d_pool[f,pi,pj] = sum_c( fc_w[c, flat_idx] * d_logit[c] ) */
    int flat_idx = f * POOL_OUT_H * POOL_OUT_W + pi * POOL_OUT_W + pj;

    double d_pool = 0.0;
    for (int c = 0; c < NUM_CLASSES; c++) {
        double d_logit = d_softmax[batchIdx * NUM_CLASSES + c]
                       - (c == label ? 1.0 : 0.0);
        int fc_w_base = FC_W_OFFSET + c * FC_INPUT_SIZE;
        d_pool += d_params[fc_w_base + flat_idx] * d_logit;
    }

    /* --- Route gradient through max-pool --- */
    int mask = d_pool_mask[batchIdx * POOL_MASK_SIZE + flat_idx];
    int mi   = mask / POOL_SIZE;
    int mj   = mask % POOL_SIZE;
    int ci   = pi * POOL_SIZE + mi;   // conv output row that held the max
    int cj   = pj * POOL_SIZE + mj;   // conv output col that held the max

    /* Multiply by ReLU derivative */
    int conv_idx = batchIdx * CONV_OUT_SIZE + f * CONV_OUT_H * CONV_OUT_W + ci * CONV_OUT_W + cj;
    double d_conv = d_pool * d_relu_grad(d_conv_out[conv_idx]);

    /* --- Accumulate conv filter gradient --- */
    const double* img = d_images[batchIdx].pixels;
    int filter_base = f * FILTER_SIZE * FILTER_SIZE;

    for (int ki = 0; ki < FILTER_SIZE; ki++) {
        for (int kj = 0; kj < FILTER_SIZE; kj++) {
            int px = (ci + ki) * INPUT_W + (cj + kj);
            atomicAdd(&d_params_grad[filter_base + ki * FILTER_SIZE + kj],
                      d_conv * img[px]);
        }
    }

    /* --- Accumulate conv bias gradient --- */
    atomicAdd(&d_params_grad[FILTERS_SIZE + f], d_conv);
}

/* -----------------------------------------------------------------------
 * Kernel 5: Apply gradients (SGD update) — one thread per parameter
 *
 * Grid/Block sized to cover PARAMS_TOTAL elements.
 * --------------------------------------------------------------------- */
__global__ void applyGradientsGPU(
    double*       d_params,
    const double* d_params_grad,
    double        learning_rate,
    int           params_total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < params_total)
        d_params[idx] -= learning_rate * d_params_grad[idx];
}

/* -----------------------------------------------------------------------
 * Host helper: pack CNNParams → flat double array
 * --------------------------------------------------------------------- */
static void packParams(const CNNParams* p, double* flat) {
    /* Conv filters */
    for (int f = 0; f < NUM_FILTERS; f++)
        for (int ki = 0; ki < FILTER_SIZE; ki++)
            for (int kj = 0; kj < FILTER_SIZE; kj++)
                flat[f * FILTER_SIZE * FILTER_SIZE + ki * FILTER_SIZE + kj] =
                    p->filters[f][ki][kj];
    /* Conv biases */
    for (int f = 0; f < NUM_FILTERS; f++)
        flat[FILTERS_SIZE + f] = p->conv_bias[f];
    /* FC weights */
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int i = 0; i < FC_INPUT_SIZE; i++)
            flat[FC_W_OFFSET + c * FC_INPUT_SIZE + i] = p->fc_weights[c][i];
    /* FC biases */
    for (int c = 0; c < NUM_CLASSES; c++)
        flat[FC_B_OFFSET + c] = p->fc_bias[c];
}

/* -----------------------------------------------------------------------
 * Host helper: unpack flat double array → CNNParams
 * --------------------------------------------------------------------- */
static void unpackParams(const double* flat, CNNParams* p) {
    for (int f = 0; f < NUM_FILTERS; f++)
        for (int ki = 0; ki < FILTER_SIZE; ki++)
            for (int kj = 0; kj < FILTER_SIZE; kj++)
                p->filters[f][ki][kj] =
                    flat[f * FILTER_SIZE * FILTER_SIZE + ki * FILTER_SIZE + kj];
    for (int f = 0; f < NUM_FILTERS; f++)
        p->conv_bias[f] = flat[FILTERS_SIZE + f];
    for (int c = 0; c < NUM_CLASSES; c++)
        for (int i = 0; i < FC_INPUT_SIZE; i++)
            p->fc_weights[c][i] = flat[FC_W_OFFSET + c * FC_INPUT_SIZE + i];
    for (int c = 0; c < NUM_CLASSES; c++)
        p->fc_bias[c] = flat[FC_B_OFFSET + c];
}

/* -----------------------------------------------------------------------
 * Macro for CUDA error checking
 * --------------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            printf("CUDA error %s:%d — %s\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(_e));                                 \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* -----------------------------------------------------------------------
 * trainNetworkBatchGPU
 * Matches the extern declaration in cnn_mnist.c
 * --------------------------------------------------------------------- */
extern "C" double trainNetworkBatchGPU(ImageVector* images, int numImages,
                                       CNNParams* params, double learning_rate)
{
    /* --- Pack parameters into a single flat array --- */
    double* h_params = (double*)malloc(PARAMS_TOTAL * sizeof(double));
    packParams(params, h_params);

    /* --- Allocate persistent GPU buffers (weights, persistent across batches) --- */
    double* d_params;
    CUDA_CHECK(cudaMalloc(&d_params, PARAMS_TOTAL * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_params, h_params, PARAMS_TOTAL * sizeof(double),
                          cudaMemcpyHostToDevice));

    /* Gradient accumulation buffer (zeroed each batch) */
    double* d_params_grad;
    CUDA_CHECK(cudaMalloc(&d_params_grad, PARAMS_TOTAL * sizeof(double)));

    /* Per-batch output buffers — sized for the largest possible batch */
    double* d_conv_out;
    double* d_pool_out;
    int*    d_pool_mask;
    double* d_logits;
    double* d_softmax;
    int*    d_labels;

    CUDA_CHECK(cudaMalloc(&d_conv_out,  BATCH_SIZE * CONV_OUT_SIZE  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pool_out,  BATCH_SIZE * FC_INPUT_SIZE  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pool_mask, BATCH_SIZE * POOL_MASK_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_logits,    BATCH_SIZE * NUM_CLASSES    * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_softmax,   BATCH_SIZE * NUM_CLASSES    * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_labels,    BATCH_SIZE                  * sizeof(int)));

    /* Batch-level image buffer */
    ImageVector* d_images;
    CUDA_CHECK(cudaMalloc(&d_images, BATCH_SIZE * sizeof(ImageVector)));

    /* Host-side softmax copy for loss computation */
    double* h_softmax = (double*)malloc(BATCH_SIZE * NUM_CLASSES * sizeof(double));
    int*    h_labels  = (int*)   malloc(BATCH_SIZE               * sizeof(int));

    double total_loss = 0.0;

    /* ----------------------------------------------------------------
     * Batch loop
     * -------------------------------------------------------------- */
    for (int i = 0; i < numImages; i += BATCH_SIZE) {
        int bs = (i + BATCH_SIZE <= numImages) ? BATCH_SIZE : (numImages - i);

        /* Upload batch */
        CUDA_CHECK(cudaMemcpy(d_images, &images[i],
                              bs * sizeof(ImageVector), cudaMemcpyHostToDevice));

        /* Extract labels to device */
        for (int b = 0; b < bs; b++) h_labels[b] = images[i + b].label;
        CUDA_CHECK(cudaMemcpy(d_labels, h_labels, bs * sizeof(int),
                              cudaMemcpyHostToDevice));

        /* Zero gradient buffer */
        CUDA_CHECK(cudaMemset(d_params_grad, 0, PARAMS_TOTAL * sizeof(double)));

        /* --- Forward: Conv + Pool --- */
        /* Block: (NUM_FILTERS, POOL_OUT_H, POOL_OUT_W) = (8, 12, 12) = 1152 threads */
        dim3 convBlock(NUM_FILTERS, POOL_OUT_H, POOL_OUT_W);
        dim3 batchGrid(bs);

        forwardConvPoolGPU<<<batchGrid, convBlock>>>(
            d_images, d_params,
            d_conv_out, d_pool_out, d_pool_mask,
            numImages, i);
        CUDA_CHECK(cudaGetLastError());

        /* --- Forward: FC + Softmax --- */
        forwardFCSoftmaxGPU<<<batchGrid, NUM_CLASSES>>>(
            d_pool_out, d_params,
            d_logits, d_softmax,
            numImages, i);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* --- Compute cross-entropy loss on host --- */
        CUDA_CHECK(cudaMemcpy(h_softmax, d_softmax,
                              bs * NUM_CLASSES * sizeof(double),
                              cudaMemcpyDeviceToHost));
        for (int b = 0; b < bs; b++) {
            double p_correct = h_softmax[b * NUM_CLASSES + h_labels[b]];
            total_loss += -log(p_correct + 1e-12);
        }

        /* --- Backward: FC gradients --- */
        backwardFCGPU<<<batchGrid, NUM_CLASSES>>>(
            d_pool_out, d_softmax, d_labels,
            d_params_grad,
            d_pool_out,   /* NOTE: d_d_pool_out reuses pool_out buffer after FC is done */
            numImages, i);
        CUDA_CHECK(cudaGetLastError());

        /* --- Backward: Conv+Pool gradients --- */
        backwardConvPoolGPU<<<batchGrid, convBlock>>>(
            d_images, d_params,
            d_softmax, d_labels,
            d_conv_out, d_pool_mask,
            d_params_grad,
            numImages, i);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        /* --- Apply gradients (SGD) --- */
        /* Scale gradients by 1/bs for the batch average */
        int applyThreads = 256;
        int applyBlocks  = (PARAMS_TOTAL + applyThreads - 1) / applyThreads;
        applyGradientsGPU<<<applyBlocks, applyThreads>>>(
            d_params, d_params_grad,
            learning_rate / bs,
            PARAMS_TOTAL);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* --- Copy updated parameters back to host --- */
    CUDA_CHECK(cudaMemcpy(h_params, d_params, PARAMS_TOTAL * sizeof(double),
                          cudaMemcpyDeviceToHost));
    unpackParams(h_params, params);

    /* --- Cleanup --- */
    cudaFree(d_params);
    cudaFree(d_params_grad);
    cudaFree(d_conv_out);
    cudaFree(d_pool_out);
    cudaFree(d_pool_mask);
    cudaFree(d_logits);
    cudaFree(d_softmax);
    cudaFree(d_labels);
    cudaFree(d_images);

    free(h_params);
    free(h_softmax);
    free(h_labels);

    return total_loss;
}