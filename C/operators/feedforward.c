#include <stdlib.h>
#include <string.h>
#include "neuron_layer.h"
#include "feedforward.h"
#include "common.h"

void layerFeedIn(neuron *n, double *input) {
  // copy input values into the neuron's own input buffer
  n->inputs = (double *)malloc(n->input_dimensions * sizeof(double));
  for (unsigned int i=0; i<n->input_dimensions; i++) {
    n->inputs[i] = input[i];
  }
}

double *matrixMultiplication(neuron *n) {
  // allocate buffer for output values
  n->outputs = (double *)malloc(n->output_dimensions * sizeof(double));
  memset(n->outputs, 0, n->output_dimensions * sizeof(double));

  // multiply weights matrix and input vector
  for (unsigned int o=0; o<n->output_dimensions; o++) {
    for (unsigned int i=0; i<n->input_dimensions; i++) {
      n->outputs[o] += indexWeightsMatrix(n->weights_matrix, i, o) * n->inputs[i];
    }
    // add bias
    n->outputs[o] += n->bias[o];
  }

  // return values
  return n->outputs;
}
