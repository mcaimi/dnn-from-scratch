#include "linear_layer.h"
#include "sigmoid_layer.h"

// TEST BENCH
int main(int argc, char **argv) {
  // Allocate a linear
  linear *first;
  first = linearCreate(8, 8, 0.000004);

  // allocate a sigmoid
  sigmoid *sig;
  sig = sigmoidCreate(8);

  // display linear info
  linearInfo(first);
  sigmoidInfo(sig);

  // display weight matrix
  printf("%s\n", "---- WEIGHTS MATRIX ----");
  displayWeights(first->weights_matrix, first->input_dimensions, first->output_dimensions);
  printf("%s\n", "---- BIAS VECTOR ----");
  printf("\n");
  for (unsigned int i=0; i < first->output_dimensions; i++) {
    printf("%f\n", first->bias[i]);
  }

  // save linear checkpoint
  linearSaveCheckpoint(first, "linearckpt.bin");

  // allocate a new linear and load checkpoint from file
  linear *loaded;
  loaded = linearLoadCheckpoint("linearckpt.bin");

  // display linear info
  linearInfo(loaded);

  // display weight matrix
  printf("%s\n", "---- WEIGHTS MATRIX ----");
  displayWeights(loaded->weights_matrix, loaded->input_dimensions, loaded->output_dimensions);
  printf("\n");
  for (unsigned int i=0; i < loaded->output_dimensions; i++) {
    printf("%f\n", loaded->bias[i]);
  }

  // save linear checkpoint
  linearSaveCheckpoint(loaded, "linearckpt1.bin");

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
  linearFeedIn(first, in);

  printf("%s\n", "---> Forward pass (matrixMultiplication)...");
  double *out = linearFeedForward(first);

  printf("%s\n", "---> Computed Output Values:");
  for (unsigned int i=0; i < first->output_dimensions; i++) {
    printf("%f\n", out[i]);
  }

  printf("%s\n", "---> Passing values to Sigmoid Layer:");
  sigmoidFeedIn(sig, out);
  printf("%s\n", "---> Forward pass (sigmoid)...");
  double *sigout = sigmoidFeedForward(sig);

  printf("%s\n", "---> Allocating random gradient vector...");
  double *grad = randomVector(8);
  double *updated_grad;
  for (unsigned int i=0; i < first->output_dimensions; i++) {
    printf("%f\n", grad[i]);
  }

  printf("%s\n", "---> Displaying Weight Matrix Before Gradient Descent...");
  displayWeights(first->weights_matrix, first->input_dimensions, first->output_dimensions);
  updated_grad = linearBackPropagation(first, grad);
  printf("%s\n", "---> Displaying Weight Matrix After Gradient Descent...");
  displayWeights(first->weights_matrix, first->input_dimensions, first->output_dimensions);
  for (unsigned int i=0; i < first->output_dimensions; i++) {
    printf("%f\n", updated_grad[i]);
  }
  printf("%s\n", "---> Displaying Sigmoid BackPropagation");
  updated_grad = sigmoidBackPropagation(sig, updated_grad);
  for (unsigned int i=0; i < sig->output_dimensions; i++) {
    printf("%f\n", updated_grad[i]);
  }


  // free resources
  linearFree(first);
  linearFree(loaded);
  sigmoidFree(sig);
  free(in);
  free(grad);
  free(updated_grad);
}
