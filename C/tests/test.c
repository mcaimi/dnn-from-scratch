#include "neuron_layer.h"
#include "feedforward.h"

// TEST BENCH
int main(int argc, char **argv) {
  // Allocate a neuron
  neuron *first;
  first = neuronCreate(8, 8, 0.000004);

  // display neuron info
  neuronInfo(first);

  // display weight matrix
  printf("%s\n", "---- WEIGHTS MATRIX ----");
  displayWeights(first->weights_matrix, first->input_dimensions, first->output_dimensions);
  printf("%s\n", "---- BIAS VECTOR ----");
  printf("\n");
  for (unsigned int i=0; i < first->output_dimensions; i++) {
    printf("%f\n", first->bias[i]);
  }

  // save neuron checkpoint
  neuronSaveCheckpoint(first, "neuronckpt.bin");

  // allocate a new neuron and load checkpoint from file
  neuron *loaded;
  loaded = neuronLoadCheckpoint("neuronckpt.bin");

  // display neuron info
  neuronInfo(loaded);

  // display weight matrix
  printf("%s\n", "---- WEIGHTS MATRIX ----");
  displayWeights(loaded->weights_matrix, loaded->input_dimensions, loaded->output_dimensions);
  printf("\n");
  for (unsigned int i=0; i < loaded->output_dimensions; i++) {
    printf("%f\n", loaded->bias[i]);
  }

  // save neuron checkpoint
  neuronSaveCheckpoint(loaded, "neuronckpt1.bin");

  // sanity check between original and disk-loaded checkpoints...
  printf("(Weights Matrix) Memory check: %d\n", compareWeightsBuffers(first->weights_matrix, loaded->weights_matrix, first->input_dimensions, first->output_dimensions));
  printf("(Bias Vector) Memory check: %d\n", compareBiasVectors(first->bias, loaded->bias, first->output_dimensions));

  // Test forward pass
  printf("%s\n", "---> Allocating random input vector...");
  double *in = randomVector(8);

  printf("%s\n", "---> Input Values:");
  printf("\n");
  for (unsigned int i=0; i < first->input_dimensions; i++) {
    printf("%f\n", in[i]);
  }
  printf("%s\n", "---> Feeding input vector to the network...");
  layerFeedIn(first, in);

  printf("%s\n", "---> Forward pass (matrixMultiplication)...");
  double *out = matrixMultiplication(first);

  printf("%s\n", "---> Computed Output Values:");
  for (unsigned int i=0; i < first->output_dimensions; i++) {
    printf("%f\n", out[i]);
  }

  // free resources
  neuronFree(first);
  neuronFree(loaded);
  free(in);
}
