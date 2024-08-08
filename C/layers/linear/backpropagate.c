#include "linear_layer.h"
#include "common.h"

double *linearBackPropagation(linear *n, double *gradient) {
  // compute gradient with respect to the inputs
  for (unsigned int i=0; i<n->input_dimensions; i++) {
    for (unsigned int o=0; o<n->output_dimensions; o++) {
      n->gradient[o] += (gradient[o] * indexWeightsMatrix(n->weights_matrix, i, o));
    }
  }

  // update weights in respect to previous layer gradient
  for (unsigned int o=0; o<n->output_dimensions; o++){
      for(unsigned int i=0; i<n->input_dimensions; i++){
        double weightvalue = indexWeightsMatrix(n->weights_matrix, i, o);
        weightvalue -= (n->learning_rate * gradient[o] * n->inputs[i]);
        (n->weights_matrix[o])[i] = weightvalue;
      }
  }

  // pass updated gradients to the next layer
  return n->gradient;
}
