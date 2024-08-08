#include "sigmoid_layer.h"
#include "common.h"

double *sigmoidBackPropagation(sigmoid *n, double *gradient) {
  // compute updated gradient vector
  for (unsigned int o=0; o<n->output_dimensions; o++) {
    n->gradient[o] = (gradient[o] * (n->outputs[o] * (1.0f - n->outputs[o])));
  }

  // pass updated gradients to the next layer
  return n->gradient;
}
