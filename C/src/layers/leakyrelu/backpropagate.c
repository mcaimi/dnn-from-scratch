#include "leakyrelu_layer.h"
#include "common.h"

double *leakyreluBackPropagation(leakyrelu *n, double *gradient) {
  // compute updated gradient vector
  for (unsigned int o=0; o<n->output_dimensions; o++) {
    // leakyrelu is not differentiable in x=0. Assume f'(x)(x=0)=alpha
    if ((n->outputs[o]) <= 0) {
      gradient[o] = n->alpha;
    } else if ((n->outputs[o]) > 0) {
      gradient[o] = 1;
    }
  }

  // pass updated gradients to the next layer
  return gradient;
}
