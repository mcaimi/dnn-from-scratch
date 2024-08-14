#include <stdlib.h>
#include "mnist.h"
#include "linear_layer.h"
#include "leakyrelu_layer.h"
#include <sys/time.h>
#include <assert.h>

// CLASSIFIER MODEL
struct __classifier_model_t {
  // linear input layer
  linear *input;
  // hidden layer
  linear *hidden;
  // output layer
  linear *output;
  // out leakyrelu layer
  leakyrelu *s_out;
};
typedef struct __classifier_model_t classifier;

// parameters
#define EPOCHS 10
#define INPUT_DIMENSIONS 768
#define HIDDEN_DIMENSIONS 32
#define OUTPUT_DIMENSIONS 10
#define ALPHA 1e-2
#define LR 4e-6

// declare model parameters
classifier *c;

// dataset
char *train_dataset_filename = "train-images-idx3-ubyte";
char *train_labels_filename = "train-labels-idx1-ubyte";
char *verification_dataset_filename = "t10k-images-idx3-ubyte";
char *verification_labels_filename = "t10k-labels-idx1-ubyte";
mnist_index *train_labels;
mnist_index *test_labels;
mnist_data *train_data;
mnist_data *test_data;

// create a new model instance
classifier *newModel(unsigned int input, unsigned int h_dim, unsigned int output) {
  // create a model
  classifier *temp = (struct __classifier_model_t *)malloc(sizeof(struct __classifier_model_t));

  // add layers
  temp->input = linearCreate(input, h_dim, LR);
  temp->hidden = linearCreate(h_dim, h_dim, LR);
  temp->output = linearCreate(h_dim, output, LR);
  temp->s_out = leakyreluCreate(output, ALPHA);

  // display network structure
  printf("Neural Network Structure:\n");
  linearInfo(temp->input);
  linearInfo(temp->hidden);
  linearInfo(temp->output);
  leakyreluInfo(temp->s_out);

  // return model object
  return temp;
}

// model feed forward function
double *modelFeedForward(classifier *c, double *input_data) {
  double *input_layer_outputs, *hidden_layer_outputs, *output_layer_outputs, *leakyrelu_response;

  // feed data into the network
  linearFeedIn(c->input, input_data);
  input_layer_outputs = linearFeedForward(c->input);

  // pass to the next layer
  linearFeedIn(c->hidden, input_layer_outputs);
  hidden_layer_outputs = linearFeedForward(c->hidden);

  // pass to the output layer
  linearFeedIn(c->output, hidden_layer_outputs);
  output_layer_outputs = linearFeedForward(c->output);

  // normalize
  leakyreluFeedIn(c->s_out, output_layer_outputs);
  leakyrelu_response = leakyreluFeedForward(c->s_out);

  // return model output vector
  return leakyrelu_response;
}

// model gradient descent
double *modelBackPropagate(classifier *c, double *gradient) {
  double *input_gradient, *hidden_gradient, *output_gradient, *leakyrelu_gradient;

  // leakyrelu
  leakyrelu_gradient = leakyreluBackPropagation(c->s_out, gradient);

  // output layer
  output_gradient = linearBackPropagation(c->output, leakyrelu_gradient);

  // hidden layer
  hidden_gradient = linearBackPropagation(c->hidden, output_gradient);
  free(output_gradient);

  // input layer
  input_gradient = linearBackPropagation(c->input, hidden_gradient);
  free(hidden_gradient);

  // return gradient
  return input_gradient;
}

// destroy model
void freeModel(classifier *m) {
  if (m->input) {
    printf("Freeing input layer...\n");
    linearFree(m->input);
  }
  if (m->hidden) {
    printf("Freeing hidden layer...\n");
    linearFree(m->hidden);
  }
  if (m->output) {
    printf("Freeing output layer...\n");
    linearFree(m->output);
  }
  if (m->s_out) {
    printf("Freeing output leakyrelu layer...\n");
    leakyreluFree(m->s_out);
  }

  // delete model
  printf("Freeing model...\n");
  free(m);
}

// training function
void train(classifier *c, mnist_data *d, mnist_index *i, unsigned int samples, unsigned int epochs) {
  // train for a specific number of epochs over the training data
  double *loss_vector;
  loss_vector = (double *)malloc(sizeof(double) * OUTPUT_DIMENSIONS);
  memset(loss_vector, 0.0f, sizeof(double) * OUTPUT_DIMENSIONS);
  for (unsigned int e=0; e<epochs; e++) {
    printf("--+== EPOCH %d STARTING... ==+--\n", e);
    for (unsigned int idx=0; idx < samples; idx++) {
      struct timeval processing_start, processing_end;

      gettimeofday(&processing_start, NULL);
      // prepare input vector
      double *frame = mnistIndexData(idx, d, TRUE);
      if (!frame) {
        printf("Cannot allocate memory for datapoint!\n");
        return;
      }

      // convert labels
      double labels[OUTPUT_DIMENSIONS];
      memset(labels, 0.0f, sizeof(double) * OUTPUT_DIMENSIONS);

      // perform feed forward operation
      double *output_vector;
      output_vector = modelFeedForward(c, frame);

      // calculate loss
      for (unsigned int n = 0; n < OUTPUT_DIMENSIONS; n++) {
        loss_vector[n] = (output_vector[n] - labels[n]);
      }
      displayWeights(&loss_vector, 1, OUTPUT_DIMENSIONS);

      // backpropagate
      double *prev_grads = modelBackPropagate(c, labels);
      free(prev_grads);
      gettimeofday(&processing_end, NULL);

      // statistics..
      unsigned long processing_time = (processing_end.tv_sec - processing_start.tv_sec);
      printf("\rTraining epoch %d/%d: Iteration: %d/%d, Processing Time %d secs, ETA: %d seconds", e, epochs, idx+1, samples, processing_time, processing_time*(samples - idx));
      fflush(stdout);

      // free leftovers
      free(frame);
    }
    printf("\n");
  }
  free(loss_vector);
}

uint8_t maxidx(double *vec, uint size) {
  assert(size > 0);

  // traverse array and find the max value
  double *found = vec;
  for (double *i = vec+1; i < (vec + size); i++) {
    if (*i > *found) {
      found = i;
    }
  }

  // return the index
  return found - vec;
}

void verify(classifier *c, mnist_data *d, mnist_index *i, unsigned int samples) {
  unsigned int accuracy = 0;

  // make predictions and compare with ground truth
  for (unsigned int idx=0; idx < samples; idx++) {
    // prepare input vector
    double *frame = mnistIndexData(idx, d, TRUE);
    if (!frame) {
      printf("Cannot allocate memory for datapoint!\n");
      return;
    }

    // perform feed forward operation
    double *output_vector;
    output_vector = modelFeedForward(c, frame);

    // compare outcome
    uint8_t target_label = i->labels[idx];
    uint8_t predicted_outcome = maxidx(output_vector, samples);
    if (predicted_outcome == target_label) {
      accuracy++;
    }

    // statistics..
    printf("\rIteration: %d/%d, Accuracy: %f", idx+1, samples, accuracy*100/samples);
    fflush(stdout);

    // free leftovers
    free(frame);
  }
  printf("\n");
}

// main loop
int main(int argc, char **argv) {
  // load resources
  printf("Loading Training Dataset from Disk...\n");
  train_data = mnistLoadData(train_dataset_filename);
  train_labels = mnistLoadIndex(train_labels_filename);
  printf("\n");

  // create the classifier model
  printf("Allocating a new Model...\n");
  classifier *m = newModel(INPUT_DIMENSIONS, HIDDEN_DIMENSIONS, OUTPUT_DIMENSIONS);
  printf("\n");

  // train the model
  printf("START TRAINING PHASE...\n");
  train(m, train_data, train_labels, 200, 8);
  //train(m, train_data, train_labels, train_data->n_items, EPOCHS);
  printf("\n");

  // free resources
  printf("Training Complete. Freeing Resources...\n");
  mnistFreeData(train_data);
  mnistFreeIndex(train_labels);
  printf("\n");

  // load test resources
  printf("Loading Verification Dataset from Disk...\n");
  test_data = mnistLoadData(verification_dataset_filename);
  test_labels = mnistLoadIndex(verification_labels_filename);
  printf("\n");

  // test model
  printf("START TEST PHASE...\n");
  verify(m, test_data, test_labels, 80);
  printf("\n");

  // free resources
  printf("Training Complete. Freeing Resources...\n");
  mnistFreeData(test_data);
  mnistFreeIndex(test_labels);
  printf("\n");

  // free model
  printf("Freeing classifier network..\n");
  freeModel(m);
}
