#include "linear_layer.h"
#include "common.h"

double *gradientDescent(neuron *n, double *gradient) {
  // updated gradients vector
  double *updated_grads;
  updated_grads = constantVector(n->output_dimensions, 0);

  // update weights in respect to previous layer gradient
  for (unsigned int o=0; o<n->output_dimensions; o++){
      for(unsigned int i=0; i<n->input_dimensions; i++){
        double weightvalue = indexWeightsMatrix(n->weights_matrix, i, o);
        weightvalue -= (n->learning_rate * gradient[o] * n->inputs[i]);
        (n->weights_matrix[o])[i] = weightvalue;
      }
  }

  // compute updated gradient vector
  for (unsigned int o=0; o<n->output_dimensions; o++) {
    for (unsigned int i=0; i<n->input_dimensions; i++) {
      updated_grads[o] += (gradient[o] * (n->weights_matrix[o])[i]);
    }
  }

  // pass updated gradients to the next layer
  return updated_grads;
}
