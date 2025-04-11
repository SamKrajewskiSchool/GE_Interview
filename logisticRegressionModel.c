#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
/* --- GLOBALS --- */
#define MAX_EPOCHS 5000
#define LEARNING_RATE 0.0001
#define MAX_ITER 15000
typedef struct {
    double pixels[784];  // Assuming 28x28 MNIST images
    int label;           // Binary label
} ImageVector;

typedef struct {
    double* hiddenActivations;
    double outputActivation;
} ActivationsOutput;

// Function Prototypes
double sigmoid(double x);
double diff_sigmoid(double x);
double* gradientDescent(ImageVector* images, int numImages);
double* computeActivations(ImageVector* images, int numImages, double* trainedWeightsAndBias);
double* NNcomputeActivations(ImageVector* images, int numImages, double** trainedWeightsAndBias1, double* trainedWeightsAndBias2);
void neuralNetwork(ImageVector* images, int numImages, double** trainedWeightsAndBias1, double* trainedWeightsAndBias2);
ImageVector* readImagesFromFile(const char* filename, int* numImages);
void shuffleImages(ImageVector* images, int numImages);

// CUDA Function Prototypes
double trainNetworkBatchGPU(ImageVector* images, int numImages, double** weights1, double* bias1, double* weights2, 
    double* bias2, double learning_rate);


int main() {
    int numImages = 0;
    int numTestImages = 0;
    ImageVector* images = readImagesFromFile("mnist_train.csv", &numImages);
    ImageVector* testImages = readImagesFromFile("mnist_test.csv", &numTestImages);

  

    if (images == NULL || testImages == NULL) {
        printf("Error reading image files\n");
        return 1;
    }
    
    printf("Loaded %d training images and %d test images\n", numImages, numTestImages);
    
    double* LGMtrainedWeightsAndBias = (double*)malloc(785 * sizeof(double));
    
    double** NNtrainedWeightsAndBias1 = (double**)malloc(785 * sizeof(double*));
    for (int i = 0; i < 785; i++) {
        NNtrainedWeightsAndBias1[i] = (double*)malloc(28 * sizeof(double));
    }
    double* NNtrainedWeightsAndBias2 = (double*)malloc(785 * sizeof(double));
    
    neuralNetwork(images, numImages, NNtrainedWeightsAndBias1, NNtrainedWeightsAndBias2);

    // Free allocated memory
    free(LGMtrainedWeightsAndBias);
    for (int i = 0; i < 785; i++) {
        free(NNtrainedWeightsAndBias1[i]);
    }
    free(NNtrainedWeightsAndBias1);
    free(NNtrainedWeightsAndBias2);
    
    free(images);
    free(testImages);
    
    return 0;
}

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of sigmoid function
double diff_sigmoid(double x) {
    return x * (1.0 - x);
}

ImageVector* readImagesFromFile(const char* filename, int* numImages) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open %s\n", filename);
        return NULL;
    }
    
    int count = 0;
    char buffer[10000];  // Assuming no line is longer than 10000 chars
    
    while (fgets(buffer, sizeof(buffer), file) != NULL) {
        count++;
    }
    
    // Rewind the file to the beginning
    rewind(file);
    
    // Allocate memory for the images
    ImageVector* images = (ImageVector*)malloc(count * sizeof(ImageVector));
    if (images == NULL) {
        printf("Memory allocation failed\n");
        fclose(file);
        return NULL;
    }
    
    // Read each line and parse it
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
            // Normalize pixel values to range [0,1]
            images[imageIndex].pixels[i] = atof(token) / 255.0;
        }
        
        imageIndex++;
    }

    fclose(file);
    *numImages = count;
    return images;
}

void shuffleImages(ImageVector* images, int numImages) {
    srand(time(NULL));
    for (int i = numImages - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        ImageVector temp = images[i];
        images[i] = images[j];
        images[j] = temp;
    }
}

void neuralNetwork(ImageVector* images, int numImages, double** trainedWeightsAndBias1, double* trainedWeightsAndBias2) {
    int inputs = 784;
    int hiddenUnits = 28;
    
    double** weight1 = (double**)malloc(inputs * sizeof(double*));
    for (int i = 0; i < inputs; i++) {
        weight1[i] = (double*)malloc(hiddenUnits * sizeof(double));
    }
    
    double* bias1 = (double*)malloc(hiddenUnits * sizeof(double));
    double* weight2 = (double*)malloc(hiddenUnits * sizeof(double));
    double bias2;
    
    srand(time(NULL));
    bias2 = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
    
    // Initialize weights and baises to random values
    for (int i = 0; i < hiddenUnits; i++) {
        bias1[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        weight2[i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        
        for (int j = 0; j < inputs; j++) {
            weight1[j][i] = 2.0 * ((double)rand() / RAND_MAX) - 1.0;
        }
    }
    
    double prevLoss = 1000000000;
    
    for (int epochs = 0; epochs < MAX_EPOCHS; epochs++) {
        // Shuffle training data
        shuffleImages(images, numImages);

        double loss = trainNetworkBatchGPU(images, numImages, weight1, bias1, weight2, &bias2, LEARNING_RATE);
        
        printf("Epoch: %d Loss: %f\n", epochs, loss / numImages);

        double lossReduction = prevLoss - loss;
        prevLoss = loss;
    }
    
    // Copy weights and biases to output parameters
    for (int i = 0; i < inputs; i++) {
        for (int j = 0; j < hiddenUnits; j++) {
            trainedWeightsAndBias1[i][j] = weight1[i][j];
        }
    }
    
    for (int i = 0; i < hiddenUnits; i++) {
        trainedWeightsAndBias2[i] = weight2[i];
        trainedWeightsAndBias1[784][i] = bias1[i];  // Store bias1 in the last row
    }
    
    trainedWeightsAndBias2[784] = bias2;  // Store bias2 in the last element
    
    // Free allocated memory
    for (int i = 0; i < inputs; i++) {
        free(weight1[i]);
    }
    free(weight1);
    free(bias1);
    free(weight2);
}
