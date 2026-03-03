#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>

/* --- GLOBALS --- */
#define MAX_EPOCHS       5000
#define LEARNING_RATE    0.0001
#define MAX_ITER         15000

/* --- CNN ARCHITECTURE --- */
#define INPUT_H          28       // Input image height
#define INPUT_W          28       // Input image width
#define NUM_FILTERS      8        // Number of conv filters
#define FILTER_SIZE      5        // 5x5 conv kernel
#define CONV_OUT_H       (INPUT_H - FILTER_SIZE + 1)   // 24
#define CONV_OUT_W       (INPUT_W - FILTER_SIZE + 1)   // 24
#define POOL_SIZE        2        // 2x2 max-pool
#define POOL_OUT_H       (CONV_OUT_H / POOL_SIZE)      // 12
#define POOL_OUT_W       (CONV_OUT_W / POOL_SIZE)      // 12
#define FC_INPUT_SIZE    (NUM_FILTERS * POOL_OUT_H * POOL_OUT_W) // 8*12*12 = 1152
#define NUM_CLASSES      10       // Digits 0-9

/* -----------------------------------------------------------------------
 * Data structures
 * --------------------------------------------------------------------- */
typedef struct {
    double pixels[784];   // 28x28 flattened MNIST image
    int    label;         // 0-9
} ImageVector;

/* Holds all learnable parameters for the CNN */
typedef struct {
    /* Conv layer: NUM_FILTERS kernels of FILTER_SIZE x FILTER_SIZE, plus bias */
    double filters[NUM_FILTERS][FILTER_SIZE][FILTER_SIZE];
    double conv_bias[NUM_FILTERS];

    /* Fully-connected layer: FC_INPUT_SIZE -> NUM_CLASSES, plus bias */
    double fc_weights[NUM_CLASSES][FC_INPUT_SIZE];
    double fc_bias[NUM_CLASSES];
} CNNParams;

/* Intermediate activations (used in forward/backward pass) */
typedef struct {
    double conv_out [NUM_FILTERS][CONV_OUT_H][CONV_OUT_W]; // pre-pool feature maps
    double pool_out [NUM_FILTERS][POOL_OUT_H][POOL_OUT_W]; // post-pool feature maps
    int    pool_mask[NUM_FILTERS][POOL_OUT_H][POOL_OUT_W]; // flat index of max location
    double fc_in    [FC_INPUT_SIZE];                        // flattened pool output
    double fc_out   [NUM_CLASSES];                          // pre-softmax logits
    double softmax  [NUM_CLASSES];                          // final probabilities
} CNNActivations;

/* -----------------------------------------------------------------------
 * Function prototypes
 * --------------------------------------------------------------------- */
/* Activation / loss */
double  relu(double x);
double  diff_relu(double x);
void    softmax(const double* logits, double* probs, int n);
double  cross_entropy_loss(const double* probs, int label, int n);

/* Forward pass */
void    conv_forward (const double image[INPUT_H][INPUT_W],
                      const CNNParams* p, CNNActivations* a);
void    pool_forward (CNNActivations* a);
void    fc_forward   (CNNActivations* a, const CNNParams* p);

/* Full forward + backward (CPU reference) */
void    forward      (const ImageVector* img, const CNNParams* p, CNNActivations* a);
void    backward     (const ImageVector* img, CNNActivations* a,
                      CNNParams* p, double lr);

/* Training */
void    cnn_train    (ImageVector* images, int numImages, CNNParams* p);
int     cnn_predict  (const ImageVector* img, const CNNParams* p);
double  evaluate     (ImageVector* images, int numImages, CNNParams* p);

/* I/O helpers (unchanged from original) */
ImageVector* readImagesFromFile(const char* filename, int* numImages);
void         shuffleImages     (ImageVector* images, int numImages);

/* CUDA extern stub (implemented externally in .cu file) */
double trainNetworkBatchGPU(ImageVector* images, int numImages,
                            CNNParams* params, double learning_rate);

/* -----------------------------------------------------------------------
 * main
 * --------------------------------------------------------------------- */
int main(void) {
    int numImages     = 0;
    int numTestImages = 0;

    ImageVector* images     = readImagesFromFile("mnist_train.csv", &numImages);
    ImageVector* testImages = readImagesFromFile("mnist_test.csv",  &numTestImages);

    if (images == NULL || testImages == NULL) {
        printf("Error reading image files\n");
        return 1;
    }
    printf("Loaded %d training images and %d test images\n", numImages, numTestImages);

    /* Allocate and initialise CNN parameters */
    CNNParams* params = (CNNParams*)malloc(sizeof(CNNParams));
    if (!params) { printf("Memory allocation failed\n"); return 1; }

    srand((unsigned)time(NULL));
    /* He-like initialisation for conv filters */
    double conv_scale = sqrt(2.0 / (FILTER_SIZE * FILTER_SIZE));
    for (int f = 0; f < NUM_FILTERS; f++) {
        params->conv_bias[f] = 0.0;
        for (int i = 0; i < FILTER_SIZE; i++)
            for (int j = 0; j < FILTER_SIZE; j++)
                params->filters[f][i][j] =
                    conv_scale * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
    }
    /* He-like initialisation for FC layer */
    double fc_scale = sqrt(2.0 / FC_INPUT_SIZE);
    for (int c = 0; c < NUM_CLASSES; c++) {
        params->fc_bias[c] = 0.0;
        for (int i = 0; i < FC_INPUT_SIZE; i++)
            params->fc_weights[c][i] =
                fc_scale * (2.0 * ((double)rand() / RAND_MAX) - 1.0);
    }

    /* Train */
    cnn_train(images, numImages, params);

    /* Evaluate on test set */
    double acc = evaluate(testImages, numTestImages, params);
    printf("Test accuracy: %.2f%%\n", acc * 100.0);

    free(params);
    free(images);
    free(testImages);
    return 0;
}

/* -----------------------------------------------------------------------
 * Activation / loss helpers
 * --------------------------------------------------------------------- */
double relu(double x)       { return x > 0.0 ? x : 0.0; }
double diff_relu(double x)  { return x > 0.0 ? 1.0 : 0.0; }

void softmax(const double* logits, double* probs, int n) {
    double max_val = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > max_val) max_val = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n; i++) { probs[i] = exp(logits[i] - max_val); sum += probs[i]; }
    for (int i = 0; i < n; i++) probs[i] /= sum;
}

double cross_entropy_loss(const double* probs, int label, int n) {
    (void)n;
    return -log(probs[label] + 1e-12);
}

/* -----------------------------------------------------------------------
 * Forward pass
 * --------------------------------------------------------------------- */
void conv_forward(const double image[INPUT_H][INPUT_W],
                  const CNNParams* p, CNNActivations* a) {
    for (int f = 0; f < NUM_FILTERS; f++)
        for (int i = 0; i < CONV_OUT_H; i++)
            for (int j = 0; j < CONV_OUT_W; j++) {
                double sum = p->conv_bias[f];
                for (int ki = 0; ki < FILTER_SIZE; ki++)
                    for (int kj = 0; kj < FILTER_SIZE; kj++)
                        sum += p->filters[f][ki][kj] * image[i + ki][j + kj];
                a->conv_out[f][i][j] = relu(sum);
            }
}

void pool_forward(CNNActivations* a) {
    for (int f = 0; f < NUM_FILTERS; f++)
        for (int i = 0; i < POOL_OUT_H; i++)
            for (int j = 0; j < POOL_OUT_W; j++) {
                int ri = i * POOL_SIZE, rj = j * POOL_SIZE;
                double max_val = -1e30;
                int    max_idx = 0;
                for (int pi = 0; pi < POOL_SIZE; pi++)
                    for (int pj = 0; pj < POOL_SIZE; pj++) {
                        double v = a->conv_out[f][ri + pi][rj + pj];
                        if (v > max_val) {
                            max_val = v;
                            max_idx = pi * POOL_SIZE + pj;
                        }
                    }
                a->pool_out [f][i][j] = max_val;
                a->pool_mask[f][i][j] = max_idx;
            }
}

void fc_forward(CNNActivations* a, const CNNParams* p) {
    /* Flatten pool_out -> fc_in */
    int idx = 0;
    for (int f = 0; f < NUM_FILTERS; f++)
        for (int i = 0; i < POOL_OUT_H; i++)
            for (int j = 0; j < POOL_OUT_W; j++)
                a->fc_in[idx++] = a->pool_out[f][i][j];

    /* FC layer: logits */
    for (int c = 0; c < NUM_CLASSES; c++) {
        double sum = p->fc_bias[c];
        for (int i = 0; i < FC_INPUT_SIZE; i++)
            sum += p->fc_weights[c][i] * a->fc_in[i];
        a->fc_out[c] = sum;
    }

    /* Softmax */
    softmax(a->fc_out, a->softmax, NUM_CLASSES);
}

void forward(const ImageVector* img, const CNNParams* p, CNNActivations* a) {
    /* Reshape flat pixels -> 2D */
    double image2d[INPUT_H][INPUT_W];
    for (int i = 0; i < INPUT_H; i++)
        for (int j = 0; j < INPUT_W; j++)
            image2d[i][j] = img->pixels[i * INPUT_W + j];

    conv_forward(image2d, p, a);
    pool_forward(a);
    fc_forward(a, p);
}

/* -----------------------------------------------------------------------
 * Backward pass (CPU reference SGD)
 * --------------------------------------------------------------------- */
void backward(const ImageVector* img, CNNActivations* a,
              CNNParams* p, double lr) {

    /* --- dL/d(logit_c) = softmax_c - 1{c == label}  (softmax + CE gradient) --- */
    double d_logit[NUM_CLASSES];
    for (int c = 0; c < NUM_CLASSES; c++)
        d_logit[c] = a->softmax[c] - (c == img->label ? 1.0 : 0.0);

    /* --- Gradients for FC layer --- */
    double d_fc_in[FC_INPUT_SIZE];
    memset(d_fc_in, 0, sizeof(d_fc_in));

    for (int c = 0; c < NUM_CLASSES; c++) {
        /* bias */
        p->fc_bias[c] -= lr * d_logit[c];
        /* weights */
        for (int i = 0; i < FC_INPUT_SIZE; i++) {
            d_fc_in[i]             += p->fc_weights[c][i] * d_logit[c];
            p->fc_weights[c][i]    -= lr * d_logit[c] * a->fc_in[i];
        }
    }

    /* --- Un-flatten d_fc_in -> d_pool_out --- */
    double d_pool_out[NUM_FILTERS][POOL_OUT_H][POOL_OUT_W];
    int idx = 0;
    for (int f = 0; f < NUM_FILTERS; f++)
        for (int i = 0; i < POOL_OUT_H; i++)
            for (int j = 0; j < POOL_OUT_W; j++)
                d_pool_out[f][i][j] = d_fc_in[idx++];

    /* --- Backprop through max-pool -> d_conv_out --- */
    double d_conv_out[NUM_FILTERS][CONV_OUT_H][CONV_OUT_W];
    memset(d_conv_out, 0, sizeof(d_conv_out));

    for (int f = 0; f < NUM_FILTERS; f++)
        for (int i = 0; i < POOL_OUT_H; i++)
            for (int j = 0; j < POOL_OUT_W; j++) {
                int mi = a->pool_mask[f][i][j] / POOL_SIZE;
                int mj = a->pool_mask[f][i][j] % POOL_SIZE;
                d_conv_out[f][i * POOL_SIZE + mi][j * POOL_SIZE + mj] =
                    d_pool_out[f][i][j];
            }

    /* Multiply by ReLU derivative */
    for (int f = 0; f < NUM_FILTERS; f++)
        for (int i = 0; i < CONV_OUT_H; i++)
            for (int j = 0; j < CONV_OUT_W; j++)
                d_conv_out[f][i][j] *= diff_relu(a->conv_out[f][i][j]);

    /* --- Gradients for conv filters --- */
    double image2d[INPUT_H][INPUT_W];
    for (int i = 0; i < INPUT_H; i++)
        for (int j = 0; j < INPUT_W; j++)
            image2d[i][j] = img->pixels[i * INPUT_W + j];

    for (int f = 0; f < NUM_FILTERS; f++) {
        double d_bias = 0.0;
        for (int i = 0; i < CONV_OUT_H; i++)
            for (int j = 0; j < CONV_OUT_W; j++)
                d_bias += d_conv_out[f][i][j];
        p->conv_bias[f] -= lr * d_bias;

        for (int ki = 0; ki < FILTER_SIZE; ki++)
            for (int kj = 0; kj < FILTER_SIZE; kj++) {
                double d_w = 0.0;
                for (int i = 0; i < CONV_OUT_H; i++)
                    for (int j = 0; j < CONV_OUT_W; j++)
                        d_w += d_conv_out[f][i][j] * image2d[i + ki][j + kj];
                p->filters[f][ki][kj] -= lr * d_w;
            }
    }
}

/* -----------------------------------------------------------------------
 * Training loop
 * --------------------------------------------------------------------- */
void cnn_train(ImageVector* images, int numImages, CNNParams* params) {
    CNNActivations act;
    double prevLoss = 1e18;

    for (int epoch = 0; epoch < MAX_EPOCHS; epoch++) {
        shuffleImages(images, numImages);

#ifdef USE_GPU
        /* --- GPU path: hand off entire batch to CUDA kernel --- */
        double loss = trainNetworkBatchGPU(images, numImages, params, LEARNING_RATE);
#else
        /* --- CPU path: SGD one sample at a time --- */
        double loss = 0.0;
        for (int n = 0; n < numImages; n++) {
            forward(&images[n], params, &act);
            loss += cross_entropy_loss(act.softmax, images[n].label, NUM_CLASSES);
            backward(&images[n], &act, params, LEARNING_RATE);
        }
#endif

        double avg_loss = loss / numImages;
        printf("Epoch %4d  |  Loss: %.6f  |  Delta: %+.6f\n",
               epoch, avg_loss, prevLoss - avg_loss);
        prevLoss = avg_loss;
    }
}

/* -----------------------------------------------------------------------
 * Inference helpers
 * --------------------------------------------------------------------- */
int cnn_predict(const ImageVector* img, const CNNParams* p) {
    CNNActivations act;
    forward(img, p, &act);
    int best = 0;
    for (int c = 1; c < NUM_CLASSES; c++)
        if (act.softmax[c] > act.softmax[best]) best = c;
    return best;
}

double evaluate(ImageVector* images, int numImages, CNNParams* p) {
    int correct = 0;
    for (int n = 0; n < numImages; n++)
        if (cnn_predict(&images[n], p) == images[n].label) correct++;
    return (double)correct / numImages;
}

/* -----------------------------------------------------------------------
 * CUDA extern stub
 * Implement this function in a separate .cu file.  It receives the full
 * parameter struct and should perform a forward + backward pass for every
 * image in the batch on the GPU, update params in-place, and return the
 * total cross-entropy loss for the batch.
 * --------------------------------------------------------------------- */
#ifndef USE_GPU
double trainNetworkBatchGPU(ImageVector* images, int numImages,
                            CNNParams* params, double learning_rate) {
    (void)images; (void)numImages; (void)params; (void)learning_rate;
    fprintf(stderr, "trainNetworkBatchGPU: compiled without USE_GPU, stub called.\n");
    return 0.0;
}
#endif

/* -----------------------------------------------------------------------
 * I/O helpers (unchanged from original)
 * --------------------------------------------------------------------- */
ImageVector* readImagesFromFile(const char* filename, int* numImages) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) { printf("Failed to open %s\n", filename); return NULL; }

    int   count  = 0;
    char  buffer[10000];
    while (fgets(buffer, sizeof(buffer), file) != NULL) count++;
    rewind(file);

    ImageVector* images = (ImageVector*)malloc(count * sizeof(ImageVector));
    if (images == NULL) {
        printf("Memory allocation failed\n");
        fclose(file);
        return NULL;
    }

    int imageIndex = 0;
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        char* token = strtok(buffer, ",");
        images[imageIndex].label = atoi(token);
        for (int i = 0; i < 784; i++) {
            token = strtok(NULL, ",");
            if (token == NULL) {
                printf("Error parsing file at image %d, pixel %d\n", imageIndex, i);
                break;
            }
            images[imageIndex].pixels[i] = atof(token) / 255.0;
        }
        imageIndex++;
    }
    fclose(file);
    *numImages = count;
    return images;
}

void shuffleImages(ImageVector* images, int numImages) {
    srand((unsigned)time(NULL));
    for (int i = numImages - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        ImageVector temp = images[i];
        images[i]        = images[j];
        images[j]        = temp;
    }
}