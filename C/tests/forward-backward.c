#include <stdio.h>
#include "common.h"
#include "linear_layer.h"
#include "sigmoid_layer.h"
#include "mnist.h"

struct {
  unsigned int input;
  unsigned int hidden;
  unsigned int output;
  double lr;
} test_cases[] = { { .input = 784, .hidden = 1536, .output = 10, .lr = 0.00004 } };

int main(int argc, char **argv) {
  if (IS_LE) {
    printf("Machine is Little Endian\n");
  } else {
    printf("Machine is Big Endian\n");
  }

  uint n_cases = sizeof(test_cases)/sizeof(test_cases[0]);

  linear *l_test_input;
  sigmoid *s_input;
  linear *l_test_hidden;
  sigmoid *s_hidden;
  linear *l_test_output;
  sigmoid *s_output;

  mnist_data *data;
  mnist_index *labels;
  data = mnistLoadData("train-images-idx3-ubyte");
  labels = mnistLoadIndex("train-labels-idx1-ubyte");

 for (unsigned int i=0; i<n_cases; i++) {
   printf("-= [Linear Layer] %s [%d] (inputs: %d, outputs: %d, lr: %f) =-\n", "Running case ", i, test_cases[i].input, test_cases[i].output, test_cases[i].lr);
   l_test_input = (linear *)linearCreate(test_cases[i].input, test_cases[i].hidden, test_cases[i].lr);
   l_test_hidden = (linear *)linearCreate(test_cases[i].hidden, test_cases[i].hidden, test_cases[i].lr);
   l_test_output = (linear *)linearCreate(test_cases[i].hidden, test_cases[i].output, test_cases[i].lr);
   printf("-= [Sigmoid Layer] %s [%d] =-\n", "Running case ", i);
   s_input = (sigmoid *)sigmoidCreate(test_cases[i].hidden);
   s_hidden = (sigmoid *)sigmoidCreate(test_cases[i].hidden);
   s_output = (sigmoid *)sigmoidCreate(test_cases[i].output);
   linearInfo(l_test_input);
   sigmoidInfo(s_input);
   linearInfo(l_test_hidden);
   sigmoidInfo(s_hidden);
   linearInfo(l_test_output);
   sigmoidInfo(s_output);

   // make a prediction
   unsigned int rand_item = (unsigned int)(rand()/data->n_items);
   double *input_vector = mnistIndexData(rand_item, data, TRUE);
   double target_vector[test_cases[i].output];
   memset(&target_vector, 0.0f, test_cases[i].output);
   target_vector[labels->labels[rand_item]] = 1.0f;


   // input
   linearFeedIn(l_test_input, input_vector);
   double *input_forward_pass = linearFeedForward(l_test_input);
   printf("(INPUT LAYER) Forward Pass Vector (0x%X):\n", input_forward_pass);
   sigmoidFeedIn(s_input, input_forward_pass);
   input_forward_pass = sigmoidFeedForward(s_input);
   displayWeights(&input_forward_pass, test_cases[i].hidden, 1);

   // hidden
   linearFeedIn(l_test_hidden, input_forward_pass);
   double *hidden_forward_pass = linearFeedForward(l_test_hidden);
   printf("(HIDDEN LAYER) Forward Pass Vector (0x%X):\n", hidden_forward_pass);
   sigmoidFeedIn(s_hidden, hidden_forward_pass);
   hidden_forward_pass = sigmoidFeedForward(s_hidden);
   displayWeights(&hidden_forward_pass, test_cases[i].hidden, 1);

   // output
   linearFeedIn(l_test_output, hidden_forward_pass);
   double *output_forward_pass = linearFeedForward(l_test_output);
   printf("(OUTPUT LAYER) Forward Pass Vector (0x%X):\n", output_forward_pass);
   sigmoidFeedIn(s_output, output_forward_pass);
   output_forward_pass = sigmoidFeedForward(s_output);
   displayWeights(&output_forward_pass, test_cases[i].output, 1);

   // simulate gradient computation
   double *rand_vector = randomVector(test_cases[i].output);
   printf("(GRADIENT DESCENT TEST) Random Vector:\n");
   displayWeights(&rand_vector, test_cases[i].output, 1);
   double *prev_grad = constantVector(test_cases[i].output, 0.0f);
   for (unsigned int n=0; n<test_cases[i].output; n++){
     double diffval = rand_vector[n] - output_forward_pass[n];
     prev_grad[n] = diffval;
   }

   // perform backpropagation
   // output
   double *so_grads = sigmoidBackPropagation(s_output, prev_grad);
   displayWeights(&so_grads, test_cases[i].output, 1);
   double *o_grads = linearBackPropagation(l_test_output, so_grads);
   printf("(OUTPUT) Backpropagation Vector:\n");
   displayWeights(&o_grads, test_cases[i].hidden, 1);

   // hidden
   double *sh_grads = sigmoidBackPropagation(s_hidden, o_grads);
   displayWeights(&sh_grads, test_cases[i].hidden, 1);
   double *h_grads = linearBackPropagation(l_test_hidden, sh_grads);
   printf("(HIDDEN) Backpropagation Vector:\n");
   displayWeights(&h_grads, test_cases[i].output, 1);

   // input
   double *si_grads = sigmoidBackPropagation(s_hidden, h_grads);
   displayWeights(&si_grads, test_cases[i].input, 1);
   double *i_grads = linearBackPropagation(l_test_input, si_grads);
   printf("(INPUT) Backpropagation Vector:\n");
   displayWeights(&i_grads, test_cases[i].input, 1);

   // free resources
   linearFree(l_test_input);
   linearFree(l_test_hidden);
   linearFree(l_test_output);
   sigmoidFree(s_input);
   sigmoidFree(s_hidden);
   sigmoidFree(s_output);
   mnistFreeData(data);
   mnistFreeIndex(labels);
   free(input_vector);
   free(rand_vector);
 }
}
