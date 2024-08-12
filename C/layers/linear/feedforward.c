#include "linear_layer.h"
#include "common.h"

void linearFeedIn(linear *n, double *input) {
  // copy input values into the linear's own input buffer
  copyVectors(input, n->inputs, n->input_dimensions);
}

double *linearFeedForward(linear *n) {
  // multiply weights matrix and input vector
  for (unsigned int o=0; o<n->output_dimensions; o++) {
    double accumulator = 0.0f;
    for (unsigned int i=0; i<n->input_dimensions; i++) {
      accumulator += (indexWeightsMatrix(n->weights_matrix, i, o) * n->inputs[i]);
    }
    // add bias
    n->outputs[o] = accumulator + n->bias[o];
  }

  // return values
  return n->outputs;
}
