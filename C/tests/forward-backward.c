#include <stdio.h>
#include "common.h"
#include "linear_layer.h"
#include "relu_layer.h"
#include "mnist.h"

#define INPUTS 768
#define HIDDEN1 128
#define HIDDEN2 128
#define OUTPUTS 10
#define LR 0.0004

int main(int argc, char **argv) {
  if (IS_LE) {
    printf("Machine is Little Endian\n");
  } else {
    printf("Machine is Big Endian\n");
  }

  linear *l_test_input;
  linear *l_test_hidden;
  linear *l_test_output;
  relu *s_output;

  mnist_data *data;
  mnist_index *labels;
  data = mnistLoadData("train-images-idx3-ubyte");
  labels = mnistLoadIndex("train-labels-idx1-ubyte");

  printf("-= [Linear Layer] (inputs: %d, outputs: %d, lr: %f) =-\n", INPUTS, OUTPUTS, LR);
  l_test_input = (linear *)linearCreate(INPUTS, HIDDEN1, LR);
  l_test_hidden = (linear *)linearCreate(HIDDEN1, HIDDEN2, LR);
  l_test_output = (linear *)linearCreate(HIDDEN2, OUTPUTS, LR);
  printf("-= [relu Layer] [%d] =-\n", OUTPUTS);
  s_output = (relu *)reluCreate(OUTPUTS);
  linearInfo(l_test_input);
  linearInfo(l_test_hidden);
  linearInfo(l_test_output);
  reluInfo(s_output);

  // make a prediction
  unsigned int rand_item = (unsigned int)(rand()/data->n_items);
  double *input_vector = mnistIndexData(rand_item, data, TRUE);
  double target_vector[OUTPUTS];
  memset(&target_vector, 0.0f, OUTPUTS);
  target_vector[labels->labels[rand_item]] = 1.0f;

  // input
  linearFeedIn(l_test_input, input_vector);
  double *input_forward_pass = linearFeedForward(l_test_input);

  // hidden
  linearFeedIn(l_test_hidden, input_forward_pass);
  double *hidden_forward_pass = linearFeedForward(l_test_hidden);

  // output
  linearFeedIn(l_test_output, hidden_forward_pass);
  double *output_forward_pass = linearFeedForward(l_test_output);
  reluFeedIn(s_output, output_forward_pass);
  double *output_relu = reluFeedForward(s_output);
  printf("Forward Pass Vector (0x%X):\n", output_relu);
  displayWeights(&output_relu, OUTPUTS, 1);

  // simulate gradient computation
  double *prev_grad = constantVector(OUTPUTS, 0.0f);
  for (unsigned int n=0; n<OUTPUTS; n++){
    double diffval = target_vector[n] - output_relu[n];
    prev_grad[n] = diffval;
  }

  // perform backpropagation
  // output
  double *so_grads = reluBackPropagation(s_output, prev_grad);
  double *o_grads = linearBackPropagation(l_test_output, so_grads);

  // hidden
  double *h_grads = linearBackPropagation(l_test_hidden, o_grads);

  // input
  double *i_grads = linearBackPropagation(l_test_input, h_grads);
  printf("Backpropagation Vector:\n");
  displayWeights(&i_grads, INPUTS, 1);

  // free resources
  linearFree(l_test_input);
  linearFree(l_test_hidden);
  linearFree(l_test_output);
  reluFree(s_output);
  mnistFreeData(data);
  mnistFreeIndex(labels);
  free(input_vector);
}
