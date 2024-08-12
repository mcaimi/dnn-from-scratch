#include <stdio.h>
#include "common.h"
#include "linear_layer.h"
#include "relu_layer.h"
#include "mnist.h"

#define INPUTS 768
#define HIDDEN1 128
#define LR 0.0004

int main(int argc, char **argv) {
  if (IS_LE) {
    printf("Machine is Little Endian\n");
  } else {
    printf("Machine is Big Endian\n");
  }

  linear *l_test_input;

  mnist_data *data;
  mnist_index *labels;
  data = mnistLoadData("train-images-idx3-ubyte");
  labels = mnistLoadIndex("train-labels-idx1-ubyte");

  l_test_input = (linear *)linearCreate(INPUTS, HIDDEN1, LR);
  linearInfo(l_test_input);

  // make a prediction
  unsigned int rand_item = (unsigned int)(rand()/data->n_items);
  double *input_vector = mnistIndexData(rand_item, data, TRUE);

  displayWeights(&input_vector, INPUTS, 1);

  // input
  linearFeedIn(l_test_input, input_vector);
  double *input_forward_pass = linearFeedForward(l_test_input);

  displayWeights(&input_forward_pass, HIDDEN1, 1);

  // free resources
  linearFree(l_test_input);
  mnistFreeData(data);
  mnistFreeIndex(labels);
  free(input_vector);
}
