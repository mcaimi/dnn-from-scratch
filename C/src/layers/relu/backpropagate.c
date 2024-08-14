#include "relu_layer.h"
#include "common.h"

double *reluBackPropagation(relu *n, double *gradient) {
  // compute updated gradient vector
  for (unsigned int o=0; o<n->output_dimensions; o++) {
    // relu is not differentiable in x=0. Assume f'(x)(x=0)=0
    // can be equally correct to assume f'(x=0) = 1
    if ((n->outputs[o]) <= 0) {
      gradient[o] = 0;
    } else if ((n->outputs[o]) > 0) {
      gradient[o] = 1;
    }
  }

  // pass updated gradients to the next layer
  return gradient;
}
