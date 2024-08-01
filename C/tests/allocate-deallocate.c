#include <stdio.h>
#include "common.h"
#include "linear_layer.h"
#include "sigmoid_layer.h"

#define CASES 6

struct {
  unsigned int input;
  unsigned int output;
  double lr;
} test_cases[] = { { .input = 16, .output = 32, .lr = 0.00004 },
                   { .input = 32, .output = 32, .lr = 0.00004 },
                   { .input = 32, .output = 64, .lr = 0.00004 },
                   { .input = 64, .output = 256, .lr = 0.00004 },
                   { .input = 256, .output = 128, .lr = 0.00004 },
                   { .input = 256, .output = 10, .lr = 0.00004 } };

int main(int argc, char **argv) {
  if (IS_LE) {
    printf("Machine is Little Endian\n");
  } else {
    printf("Machine is Big Endian\n");
  }

  linear *l_test;
  sigmoid *s_test;

 for (unsigned int i=0; i<CASES; i++) {
   printf("-= [Linear Layer] %s [%d] =-\n", "Running case ", i);
   l_test = (linear *)linearCreate(test_cases[i].input, test_cases[i].output, test_cases[i].lr);
   linearInfo(l_test);
   linearFree(l_test);
 }

 for (unsigned int i=0; i<CASES; i++) {
   printf("-= [Sigmoid Layer] %s [%d] =-\n", "Running case ", i);
   s_test = (sigmoid *)sigmoidCreate(test_cases[i].output);
   sigmoidInfo(s_test);
   sigmoidFree(s_test);
 }
}
