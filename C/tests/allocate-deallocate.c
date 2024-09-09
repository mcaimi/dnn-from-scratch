#include <stdio.h>
#include "linear_layer.h"

#define CASES 6

struct {
  unsigned int input;
  unsigned int output;
  double lr;
  char *filename;
} test_cases[] = { { .input = 16, .output = 32, .lr = 0.00004, .filename = "save1.ckpt"},
                   { .input = 32, .output = 32, .lr = 0.0000, .filename = "save2.ckpt" },
                   { .input = 32, .output = 64, .lr = 0.0000, .filename = "save3.ckpt" },
                   { .input = 64, .output = 256, .lr = 0.0000, .filename = "save4.ckpt" },
                   { .input = 256, .output = 128, .lr = 0.0000, .filename = "save5.ckpt" },
                   { .input = 256, .output = 10, .lr = 0.0000, .filename = "save6.ckpt" } };

int verifyWeights(linear *n) {
  for (unsigned int o=0; o<n->output_dimensions; o++) {
    for (unsigned int i=0; i<n->input_dimensions; i++) {
      double item = indexWeightsMatrix(n->weights_matrix, i, o);
      if ((item < 0) || (item > 1)) {
        printf("Fatal: item at position %d/%d is not bounded between [0,1].\n", i,o);
        return -1;
      }
    }
  }
  return 1;
}

int main(int argc, char **argv) {
  if (IS_LE) {
    printf("Machine is Little Endian\n");
  } else {
    printf("Machine is Big Endian\n");
  }

  linear *l_test;

 for (unsigned int i=0; i<CASES; i++) {
   printf("-= [Linear Layer] %s [%d] =-\n", "Running case ", i);
   l_test = (linear *)linearCreate(test_cases[i].input, test_cases[i].output, test_cases[i].lr);
   linearInfo(l_test);
   if (!verifyWeights(l_test)) printf("Linear Layer Corruption.\n");
   linearSaveCheckpoint(l_test, test_cases[i].filename);
   linearFree(l_test);
   l_test = linearLoadCheckpoint(test_cases[i].filename);
   linearInfo(l_test);
   if (!verifyWeights(l_test)) printf("Loaded Layer Corruption.\n");
   linearFree(l_test);
 }
}
