#include <math.h>
#include "leakyrelu_layer.h"
#include "common.h"

void leakyreluFeedIn(leakyrelu *n, double *input) {
  // copy input values into the leakyrelu's own input buffer
  copyVectors(input, n->inputs, n->input_dimensions);
}

double *leakyreluFeedForward(leakyrelu *n) {
  // calculate leakyrelu
  for (unsigned int o=0; o<n->output_dimensions; o++) {
    // for x<=0 => leakyrelu == alpha * x
    // for x>0 => leakyrelu == x
    if ((n->inputs[o]) > 0) {
      n->outputs[o] = n->inputs[o];
    } else {
      n->outputs[o] = n->alpha * fabs(n->inputs[o]);
    }
  }

  // return values
  return n->outputs;
}
