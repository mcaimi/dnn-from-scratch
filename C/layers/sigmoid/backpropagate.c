#include "sigmoid_layer.h"
#include "common.h"

double *sigmoidBackPropagation(sigmoid *n, double *gradient) {
  // updated gradients vector
  double *updated_grads;
  updated_grads = constantVector(n->output_dimensions, 0.0f);

  // compute updated gradient vector
  for (unsigned int o=0; o<n->output_dimensions; o++) {
    updated_grads[o] = (gradient[o] * (n->outputs[o] * (1.0f - n->outputs[o])));
  }

  // pass updated gradients to the next layer
  return updated_grads;
}
