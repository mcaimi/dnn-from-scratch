#include <stdlib.h>
#include "mnist.h"
#include "linear_layer.h"
#include "sigmoid_layer.h"
#include <sys/time.h>
#include <assert.h>

// CLASSIFIER MODEL
struct __classifier_model_t {
  // linear input layer
  linear *input;
  // input sigmoid layer
  sigmoid *s_input;
  // hidden layer
  linear *hidden;
  // hidden sigmoid layer
  sigmoid *s_hidden;
  // output layer
  linear *output;
  // out sigmoid layer
  sigmoid *s_out;
};
typedef struct __classifier_model_t classifier;

// parameters
#define EPOCHS 10
#define OUTPUT_DIMENSIONS 10
#define HIDDEN_MULTIPLIER 2

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
  temp->input = linearCreate(input, h_dim, 0.00004);
  temp->s_input = sigmoidCreate(h_dim);
  temp->hidden = linearCreate(h_dim, h_dim, 0.00004);
  temp->s_hidden = sigmoidCreate(h_dim);
  temp->output = linearCreate(h_dim, output, 0.00004);
  temp->s_out = sigmoidCreate(output);

  // display network structure
  printf("Neural Network Structure:\n");
  linearInfo(temp->input);
  sigmoidInfo(temp->s_input);
  linearInfo(temp->hidden);
  sigmoidInfo(temp->s_hidden);
  linearInfo(temp->output);
  sigmoidInfo(temp->s_out);

  // return model object
  return temp;
}

// model feed forward function
double *modelFeedForward(classifier *c, double *input_data) {
  double *input_layer_outputs, *hidden_layer_outputs, *output_layer_outputs, *sigmoid_layer_outputs;

  // feed data into the network
  linearFeedIn(c->input, input_data);
  sigmoidFeedIn(c->s_input, linearFeedForward(c->input));
  input_layer_outputs = sigmoidFeedForward(c->s_input);

  // pass to the next layer
  linearFeedIn(c->hidden, input_layer_outputs);
  sigmoidFeedIn(c->s_hidden, linearFeedForward(c->hidden));
  hidden_layer_outputs = sigmoidFeedForward(c->s_hidden);

  // pass to the output layer
  linearFeedIn(c->output, hidden_layer_outputs);
  sigmoidFeedIn(c->s_out, linearFeedForward(c->output));
  output_layer_outputs = sigmoidFeedForward(c->s_out);

  // return model output vector
  return sigmoid_layer_outputs;
}

// model gradient descent
double *modelBackPropagate(classifier *c, double *gradient) {
  double *input_gradient, *hidden_gradient, *output_gradient, *sigmoid_gradient;

  // backpropagate through the network
  sigmoid_gradient = sigmoidBackPropagation(c->s_out, gradient);

  // output layer
  output_gradient = linearBackPropagation(c->output, sigmoid_gradient);

  // hidden layer
  hidden_gradient = linearBackPropagation(c->hidden, output_gradient);

  // input layer
  input_gradient = linearBackPropagation(c->input, hidden_gradient);

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
  if (m->s_input) {
    printf("Freeing input sigmoid layer...\n");
    sigmoidFree(m->s_input);
  }
  if (m->s_hidden) {
    printf("Freeing hidden sigmoid layer...\n");
    sigmoidFree(m->s_hidden);
  }
  if (m->s_out) {
    printf("Freeing output sigmoid layer...\n");
    sigmoidFree(m->s_out);
  }

  // delete model
  printf("Freeing model...\n");
  free(m);
}

// training function
void train(classifier *c, mnist_data *d, mnist_index *i, unsigned int samples, unsigned int epochs) {
  // train for a specific number of epochs over the training data
  for (unsigned int e=0; e<epochs; e++) {
    printf("--+== EPOCH %d STARTING... ==+--\n", e);
    double *loss_vector;
    loss_vector = (double *)malloc(sizeof(double) * OUTPUT_DIMENSIONS);
    for (unsigned int idx=0; idx < samples; idx++) {
      struct timeval processing_start, processing_end;

      gettimeofday(&processing_start, NULL);
      // prepare input vector
      double *frame = mnistIndexData(idx, d, FALSE);
      if (!frame) {
        printf("Cannot allocate memory for datapoint!\n");
        return;
      }

      // convert labels
      double labels[OUTPUT_DIMENSIONS];
      memset(labels, 0.0f, OUTPUT_DIMENSIONS);
      labels[i->labels[idx]] = 1.0f;

      // perform feed forward operation
      double *output_vector;
      output_vector = modelFeedForward(c, frame);
      displayWeights(&output_vector, OUTPUT_DIMENSIONS, 1);

      // calculate loss
      for (unsigned int n = 0; n < OUTPUT_DIMENSIONS; n++) {
        loss_vector[n] = labels[n] - output_vector[n];
      }

      // backpropagate
      double *prev_grads = modelBackPropagate(c, loss_vector);
      copyVectors(prev_grads, loss_vector, OUTPUT_DIMENSIONS);
      gettimeofday(&processing_end, NULL);

      // statistics..
      unsigned long processing_time = (processing_end.tv_sec - processing_start.tv_sec);
      printf("\rTraining epoch %d/%d: Iteration: %d/%d, Processing Time %d secs, ETA: %d seconds", e, epochs, idx+1, samples, processing_time, processing_time*(samples - idx));
      fflush(stdout);

      // free leftovers
      free(frame);
    }
    free(loss_vector);
    printf("\n");
  }
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
  classifier *m = newModel((train_data->n_rows * train_data->n_cols), HIDDEN_MULTIPLIER*(train_data->n_rows * train_data->n_cols), OUTPUT_DIMENSIONS);
  printf("\n");

  // train the model
  //train(m, train_data, train_labels, EPOCHS);
  printf("START TRAINING PHASE...\n");
  train(m, train_data, train_labels, 20, 1);
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
  //verify(m, test_data, test_labels, 80);
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
