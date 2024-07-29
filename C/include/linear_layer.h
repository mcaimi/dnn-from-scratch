#include "common.h"

struct __neuron_t {
  // layer input and output dimensions
  unsigned int input_dimensions;
  unsigned int output_dimensions;

  // input and output values
  double *inputs;
  double *outputs;

  // i*o matrix
  double **weights_matrix;
  double *bias;

  // learning rate
  double learning_rate;
};
typedef struct __neuron_t neuron;

// create new neuron
neuron *neuronCreate(unsigned int, unsigned int, double);
// free neuron
void neuronFree(neuron *);
// display info
void neuronInfo(neuron *);

// set learning rate
void neuronSetLR(neuron *, double);
// save neuron checkpoint
void neuronSaveCheckpoint(neuron *, char *);
// load neuron checkpoint
neuron *neuronLoadCheckpoint(char *);

// set input values into the neuron layer
void layerFeedIn(neuron *, double *);

// feed-forward matrix multiplication
double *forwardMultiplication(neuron *);

// gradient descent
double *gradientDescent(neuron *, double *);

