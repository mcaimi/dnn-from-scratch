#include "leakyrelu_layer.h"

// allocate dynamic memory for a leakyrelu layer
int __leakyreluMemAllocate(leakyrelu *n) {
  n->inputs = (double *)malloc(n->input_dimensions * sizeof(double));
  n->outputs = (double *)malloc(n->output_dimensions * sizeof(double));
  if (!n->inputs || !n->outputs) {
    leakyreluFree(n);
    return 1;
  }

  return 0;
}

// create new leakyrelu layer
leakyrelu *leakyreluCreate(unsigned int dimensions, double parameter) {
  leakyrelu *temp;
  temp = (leakyrelu *)malloc(sizeof(struct __leakyrelu_t));
  if (!temp) return NULL;

  // initialize
  temp->input_dimensions = dimensions;
  temp->output_dimensions = dimensions;

  // sane defaults
  if (__leakyreluMemAllocate(temp) > 0) {
    return NULL;
  }

  // zero buffers for input and output
  memset(temp->outputs, 0, temp->output_dimensions * sizeof(double));
  memset(temp->inputs, 0, temp->input_dimensions * sizeof(double));

  // set leaky relu parameter
  temp->alpha = parameter;

  // return leakyrelu layer
  return temp;
}

// free leakyrelu layer
void leakyreluFree(leakyrelu *n) {
  if (n != NULL) {
    // free memory...
    if (n->inputs) {
      free(n->inputs);
    }
    if (n->outputs) {
      free(n->outputs);
    }

    // destroy leakyrelu layer
    free(n);
  }
}

void leakyreluInfo(leakyrelu *n) {
  printf("leakyrelu Layer Configuration:\n\tInput Size: %d\n\tOutput Size: %d\n\tAlpha: %e\n", n->input_dimensions, n->output_dimensions, n->alpha);
}

