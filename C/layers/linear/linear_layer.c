#include "linear_layer.h"

// allocate dynamic memory for a linear layer
int __linearMemAllocate(linear *n) {
  n->inputs = (double *)malloc(n->input_dimensions * sizeof(double));
  n->outputs = (double *)malloc(n->output_dimensions * sizeof(double));
  n->weights_matrix = randomMatrix(n->input_dimensions, n->output_dimensions);
  if (!n->inputs || !n->outputs || !n->weights_matrix) {
    linearFree(n);
    return 1;
  }

  // generate random bias vector
  n->bias = constantVector(n->output_dimensions, 0.0f);
  if (!n->bias) {
    linearFree(n);
    return 1;
  }

  return 0;
}

// create new linear layer
linear *linearCreate(unsigned int inputs, unsigned int outputs, double learning_rate) {
  linear *temp;
  temp = (linear *)malloc(sizeof(struct __linear_t));
  if (!temp) return NULL;

  // initialize
  temp->input_dimensions = inputs;
  temp->output_dimensions = outputs;

  // sane defaults
  if (__linearMemAllocate(temp) > 0) {
    return NULL;
  }

  // zero buffers for input and output
  memset(temp->outputs, 0, temp->output_dimensions * sizeof(double));
  memset(temp->inputs, 0, temp->input_dimensions * sizeof(double));

  // set learning rate
  linearSetLR(temp, learning_rate);

  // return linear layer
  return temp;
}

// free linear layer
void linearFree(linear *n) {
  if (n != NULL) {
    // free memory...
    if (n->inputs) {
      free(n->inputs);
    }
    if (n->outputs) {
      free(n->outputs);
    }

    // free weights matrix
    for (unsigned int o=(n->output_dimensions -1); o > 0; o--) {
      if (n->weights_matrix[o]) {
        free(n->weights_matrix[o]);
      }
    }

    if (n->bias) {
      free(n->bias);
    }

    // destroy linear layer
    free(n);
  }
}

// dump linear layer info
void linearInfo(linear *n) {
  printf("Linear Layer Configuration:\n\tInput Size: %d\n\tOutput Size: %d\n\tWeights Matrix: %lu bytes @ 0x%lX\n\tBias Vector: %lu bytes @ 0x%lX\n",
      n->input_dimensions, n->output_dimensions, (n->input_dimensions*n->output_dimensions*sizeof(double)), (unsigned long)(n->weights_matrix),
      (n->output_dimensions*sizeof(double)), (unsigned long)(n->bias));
  printf("Learning Rate: %f\n", n->learning_rate);
}

// set learning rate for this linear layer
void linearSetLR(linear *n, double lr) {
  if (lr < 0) {
    n->learning_rate = 0;
  } else {
    n->learning_rate = lr;
  }
}

// save the current linear state in a persistent checkpoint file
void linearSaveCheckpoint(linear *n, char *filename) {
  FILE *saveFileDesc;

  // open file for writing...
  saveFileDesc = (FILE *)fopen(filename, "w");
  if (saveFileDesc == NULL) {
    printf("Error opening checkpoint %s for writing.\n", filename);
    return;
  }

  // 1 - save linear structure
  printf("%s", "Saving structure...\n");
  fwrite(&n->input_dimensions, sizeof(unsigned int), 1, saveFileDesc);
  fwrite(&n->output_dimensions, sizeof(unsigned int), 1, saveFileDesc);
  fwrite(&n->learning_rate, sizeof(double), 1, saveFileDesc);

  // 2- save weights matrix
  printf("%s", "Saving Weights Matrix...");
  for (unsigned int i=0; i < n->output_dimensions; i++) {
    printf(" %d...", i);
    fwrite(n->weights_matrix[i], sizeof(double), n->input_dimensions, saveFileDesc);
  }
  printf("\n");

  // 3 - save bias vector
  printf("%s", "Saving Bias Vector...\n");
  fwrite(n->bias, sizeof(double), n->output_dimensions, saveFileDesc);

  // sync to disk and close descriptor
  fflush(saveFileDesc);
  fclose(saveFileDesc);
  printf("Checkpoint [%s] Saved\n", filename);
}

// create a new linear layer loading data from a checkpoint
linear *linearLoadCheckpoint(char *filename) {
  FILE *loadFileDesc;
  linear *temp;

  // open checkpoint file for reading
  printf("Loading checkpoint [%s]...\n", filename);
  loadFileDesc = (FILE *)fopen(filename, "r");
  if (loadFileDesc == NULL) {
    printf("Error opening checkpoint %s for reading.\n", filename);
    return NULL;
  }

  // 1 - load linear structure
  printf("Allocating and loading linear structure...\n");
  temp = (linear *)malloc(sizeof(struct __linear_t));
  fread(&temp->input_dimensions, sizeof(unsigned int), 1, loadFileDesc);
  fread(&temp->output_dimensions, sizeof(unsigned int), 1, loadFileDesc);
  fread(&temp->learning_rate, sizeof(double), 1, loadFileDesc);

  // 2 - load weights matrix values
  printf("Loading Weights Matrix...\n");
  temp->weights_matrix = (double **)malloc(temp->output_dimensions * sizeof(double *));
  for (unsigned int i=0; i < temp->output_dimensions; i++) {
    printf(" %d...", i);
    temp->weights_matrix[i] = (double *)malloc(temp->input_dimensions * sizeof(double));
    fread(temp->weights_matrix[i], sizeof(double), temp->input_dimensions, loadFileDesc);
  }
  printf("\n");

  // 3 - load bias vector
  printf("%s", "Loading Bias Vector...\n");
  temp->bias = (double *)malloc(temp->output_dimensions * sizeof(double));
  fread(temp->bias, sizeof(double), temp->output_dimensions, loadFileDesc);

  // finalize structure
  printf("%s", "Finalizing...\n");
  temp->inputs = (double *)malloc(temp->input_dimensions * sizeof(double));
  temp->outputs = (double *)malloc(temp->output_dimensions * sizeof(double));

  // close and return
  fclose(loadFileDesc);
  printf("Checkpoint loaded.\n");
  return temp;
}
