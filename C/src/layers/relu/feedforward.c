#include <math.h>
#include "relu_layer.h"
#include "common.h"

void reluFeedIn(relu *n, double *input) {
  // copy input values into the relu's own input buffer
  copyVectors(input, n->inputs, n->input_dimensions);
}

double *reluFeedForward(relu *n) {
  // calculate relu
  for (unsigned int o=0; o<n->output_dimensions; o++) {
    // relu => ReLU(x) = max(0,x)
    // for x<=0 => ReLU == 0
    // for x>0 => ReLU == x
    if (n->inputs[o] > 0) {
      n->outputs[o] = n->inputs[o];
    } else {
      n->outputs[o] = 0;
    }
  }

  // return values
  return n->outputs;
}
