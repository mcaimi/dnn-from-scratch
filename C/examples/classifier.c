#include <stdlib.h>
#include "linear_layer.h"
#include "sigmoid_layer.h"

// CLASSIFIER MODEL
struct __classifier_model_t {
  // linear input layer
  linear *input;
  // hidden layer
  linear *hidden;
  // output layer
  linear *output;
  // sigmoid layer
  sigmoid *sig;
};
typedef struct __classifier_model_t classifier;

// parameters
#define INPUT_DIMENSIONS 64
#define HIDDEN_DIMENSIONS 256
#define OUTPUT_DIMENSIONS 10

// declare model parameters
classifier *c;

// create a new model instance
classifier *newModel(unsigned int input, unsigned int h_dim, unsigned int output) {
  // create a model
  classifier *temp = (struct __classifier_model_t *)malloc(sizeof(struct __classifier_model_t));

  // add layers
  temp->input = linearCreate(input, h_dim, 0.00004);
  temp->hidden = linearCreate(h_dim, h_dim, 0.00004);
  temp->output = linearCreate(h_dim, output, 0.00004);
  temp->sig = sigmoidCreate(output);

  // display network structure
  printf("Neural Network Structure:\n");
  linearInfo(temp->input);
  linearInfo(temp->hidden);
  linearInfo(temp->output);
  sigmoidInfo(temp->sig);

  // return model object
  return temp;
}

// destroy model
void freeModel(classifier *m) {
  if (m->input) {
    printf("Freeing layer...\n");
    linearFree(m->input);
  }
  if (m->hidden) {
    printf("Freeing layer...\n");
    linearFree(m->hidden);
  }
  if (m->output) {
    printf("Freeing layer...\n");
    linearFree(m->output);
  }
  if (m->sig) {
    printf("Freeing layer...\n");
    sigmoidFree(m->sig);
  }

  // delete model
  printf("Freeing model...\n");
  free(m);
}

// main loop
int main(int argc, char **argv) {
  classifier *m = newModel(INPUT_DIMENSIONS, HIDDEN_DIMENSIONS, OUTPUT_DIMENSIONS);

  freeModel(m);
}
