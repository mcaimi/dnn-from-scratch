#include "linear_layer.h"
#include "common.h"

double *linearBackPropagation(linear *n, double *gradient) {
  // compute gradient with respect to the inputs
  double *prev_grad = constantVector(n->input_dimensions, 0.0);
  for (unsigned int i=0; i<n->input_dimensions; i++) {
    double accumulator = 0.0;
    for (unsigned int o=0; o<n->output_dimensions; o++) {
      accumulator += (gradient[o] * indexWeightsMatrix(n->weights_matrix, i, o));
    }
    prev_grad[i] = accumulator;
  }

  // update weights in respect to previous layer gradient
  for (unsigned int o=0; o<n->output_dimensions; o++){
      for(unsigned int i=0; i<n->input_dimensions; i++){
        (n->weights_matrix[i])[o] -= (n->learning_rate * gradient[o] * n->inputs[i]);
      }
      n->bias[o] -= n->learning_rate * gradient[o];
  }

  // pass updated gradients to the next layer
  return prev_grad;
}
