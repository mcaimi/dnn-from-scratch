#include "leakyrelu_layer.h"
#include "common.h"

double *leakyreluBackPropagation(leakyrelu *n, double *gradient) {
  // compute updated gradient vector
  for (unsigned int o=0; o<n->output_dimensions; o++) {
    // leakyrelu is not differentiable in x=0. Assume f'(x)(x=0)=alpha
    // Multiply incoming gradient by derivative of leaky ReLU
    if ((n->outputs[o]) <= 0) {
      gradient[o] *= n->alpha;
    }
    // If output > 0, gradient passes through unchanged (multiply by 1)
    // No need for else clause as gradient[o] already has correct value
  }

  // pass updated gradients to the next layer
  return gradient;
}
