#include "sigmoid_layer.h"

// allocate dynamic memory for a sigmoid layer
int __sigmoidMemAllocate(sigmoid *n) {
  n->inputs = (double *)malloc(n->input_dimensions * sizeof(double));
  n->outputs = (double *)malloc(n->output_dimensions * sizeof(double));
  if (!n->inputs || !n->outputs) {
    sigmoidFree(n);
    return 1;
  }

  return 0;
}

// create new sigmoid layer
sigmoid *sigmoidCreate(unsigned int dimensions) {
  sigmoid *temp;
  temp = (sigmoid *)malloc(sizeof(struct __sigmoid_t));
  if (!temp) return NULL;

  // initialize
  temp->input_dimensions = dimensions;
  temp->output_dimensions = dimensions;

  // sane defaults
  if (__sigmoidMemAllocate(temp) > 0) {
    return NULL;
  }

  // zero buffers for input and output
  memset(temp->outputs, 0, temp->output_dimensions * sizeof(double));
  memset(temp->inputs, 0, temp->input_dimensions * sizeof(double));

  // return sigmoid layer
  return temp;
}

// free sigmoid layer
void sigmoidFree(sigmoid *n) {
  if (n != NULL) {
    // free memory...
    if (n->inputs) {
      free(n->inputs);
    }
    if (n->outputs) {
      free(n->outputs);
    }

    // destroy sigmoid layer
    free(n);
  }
}

void sigmoidInfo(sigmoid *n) {
  printf("Sigmoid Layer Configuration:\n\tInput Size: %d\n\tOutput Size: %d\n", n->input_dimensions, n->output_dimensions);
}

