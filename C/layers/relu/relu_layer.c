#include "relu_layer.h"

// allocate dynamic memory for a relu layer
int __reluMemAllocate(relu *n) {
  n->inputs = (double *)malloc(n->input_dimensions * sizeof(double));
  n->outputs = (double *)malloc(n->output_dimensions * sizeof(double));
  if (!n->inputs || !n->outputs) {
    reluFree(n);
    return 1;
  }

  return 0;
}

// create new relu layer
relu *reluCreate(unsigned int dimensions) {
  relu *temp;
  temp = (relu *)malloc(sizeof(struct __relu_t));
  if (!temp) return NULL;

  // initialize
  temp->input_dimensions = dimensions;
  temp->output_dimensions = dimensions;

  // sane defaults
  if (__reluMemAllocate(temp) > 0) {
    return NULL;
  }

  // zero buffers for input and output
  memset(temp->outputs, 0, temp->output_dimensions * sizeof(double));
  memset(temp->inputs, 0, temp->input_dimensions * sizeof(double));

  // return relu layer
  return temp;
}

// free relu layer
void reluFree(relu *n) {
  if (n != NULL) {
    // free memory...
    if (n->inputs) {
      free(n->inputs);
    }
    if (n->outputs) {
      free(n->outputs);
    }

    // destroy relu layer
    free(n);
  }
}

void reluInfo(relu *n) {
  printf("ReLU Layer Configuration:\n\tInput Size: %d\n\tOutput Size: %d\n", n->input_dimensions, n->output_dimensions);
}

