#include <stdio.h>
#include "common.h"
#include "matrix.h"
#include "linear_layer.h"
#include "relu_layer.h"
#include "sigmoid_layer.h"
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
  relu *r_input;
  linear *l_test_hidden;
  relu *r_hidden;
  linear *l_test_output;
  relu *s_output;

  mnist_data *data;
  mnist_index *labels;
  data = mnistLoadData("train-images-idx3-ubyte");
  labels = mnistLoadIndex("train-labels-idx1-ubyte");

  printf("-= [Linear Layer] (inputs: %d, outputs: %d, lr: %f) =-\n", INPUTS, OUTPUTS, LR);
  l_test_input = (linear *)linearCreate(INPUTS, HIDDEN1, LR, FALSE);
  l_test_hidden = (linear *)linearCreate(HIDDEN1, HIDDEN2, LR, FALSE);
  l_test_output = (linear *)linearCreate(HIDDEN2, OUTPUTS, LR, FALSE);
  printf("-= [relu Layer] [%d] =-\n", HIDDEN1);
  r_input = (relu *)reluCreate(HIDDEN1);
  r_hidden = (relu *)reluCreate(HIDDEN2);
  printf("-= [sigmoid Layer] [%d] =-\n", OUTPUTS);
  s_output = (relu *)sigmoidCreate(OUTPUTS);
  linearInfo(l_test_input);
  linearInfo(l_test_hidden);
  linearInfo(l_test_output);
  reluInfo(r_input);
  reluInfo(r_hidden);
  reluInfo(s_output);

  // make a prediction
  unsigned int rand_item = (unsigned int)(rand()/data->n_items);
  double *input_vector = mnistIndexData(rand_item, data, TRUE);
  double target_vector[OUTPUTS];
  memset(&target_vector, 0.0f, sizeof(double)*OUTPUTS);
  target_vector[labels->labels[rand_item]] = 1.0f;

  // input
  linearFeedIn(l_test_input, input_vector);
  reluFeedIn(r_input, linearFeedForward(l_test_input));
  double *input_forward_pass = reluFeedForward(r_input);

  // hidden
  linearFeedIn(l_test_hidden, input_forward_pass);
  reluFeedIn(r_hidden, linearFeedForward(l_test_hidden));
  double *hidden_forward_pass = reluFeedForward(r_hidden);

  // output
  linearFeedIn(l_test_output, hidden_forward_pass);
  reluFeedIn(s_output, linearFeedForward(l_test_output));
  double *output_forward_pass = reluFeedForward(s_output);

  printf("--> [FORWARD PASS] : Out Vector:\n");
  displayWeights(&output_forward_pass, 1, OUTPUTS);

  // simulate gradient computation
  double *prev_grad = constantVector(OUTPUTS, 0.0f);
  for (unsigned int n=0; n<OUTPUTS; n++){
    double diffval = output_forward_pass[n] - target_vector[n];
    prev_grad[n] = diffval;
  }

  // perform backpropagation
  // relu
  double *so_grads = reluBackPropagation(s_output, prev_grad);

  // output
  double *o_grads = linearBackPropagation(l_test_output, so_grads);

  // hidden
  double *sh_grads = reluBackPropagation(r_hidden, o_grads);
  double *h_grads = linearBackPropagation(l_test_hidden, sh_grads);
  free(o_grads);

  // input
  double *si_grads = reluBackPropagation(r_input, h_grads);
  double *i_grads = linearBackPropagation(l_test_input, si_grads);
  free(h_grads);

  printf("--> [BACKPROPAGATION] : Out Vector:\n");
  displayWeights(&i_grads, 1, INPUTS);

  // free resources
  linearFree(l_test_input);
  linearFree(l_test_hidden);
  linearFree(l_test_output);
  reluFree(r_input);
  reluFree(r_hidden);
  reluFree(s_output);
  mnistFreeData(data);
  mnistFreeIndex(labels);
  free(input_vector);
  free(i_grads);
}
