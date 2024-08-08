#include "common.h"

// initialize random weights in memory
double **randomMatrix(unsigned int x_dim, unsigned int y_dim) {
  double **placeholder = NULL;

  // allocate space for the weight matrix
  // weights are arranged in a WxH matrix
  // values are double
  placeholder = (double **)malloc(y_dim * sizeof(double *));
  if (!placeholder) return NULL;

  // fill matrix with random numbers
  for (unsigned int y=0; y < y_dim; y++) {
    placeholder[y] = randomVector(x_dim);
  }

  // return data
  return placeholder;
}

// initialize random biases in memory
double *randomVector(unsigned int dim) {
  double *placeholder = NULL;

  // one-dimensional vector of double values
  placeholder = (double *)malloc(dim * sizeof(double));
  if (!placeholder) return NULL;

  // fill vector with random numbers
  for (unsigned int i=0; i < dim; i++) {
      double pl = (double)((double)arc4random()/(double)UINT32_MAX) *2 - 1;
      placeholder[i] = pl;
  }

  return placeholder;
}

// initialize a constant vector
double *constantVector(unsigned int dim, double const_val) {
  double *placeholder = NULL;

  // allocate buffer
  placeholder = (double *)malloc(dim * sizeof(double));
  if (!placeholder) return NULL;

  // fill vector with random numbers
  for (unsigned int i=0; i < dim; i++) {
      placeholder[i] = const_val;
  }

  return placeholder;

}

// copy vectors
void copyVectors(double *src, double *dst, unsigned int size) {
  // copy values from src to dst
  for (unsigned int i=0; i<size; i++) {
    dst[i] = src[i];
  }
}

// compare weight buffers
int compareWeightsBuffers(double **buffer_a, double **buffer_b, unsigned int x_dim, unsigned int y_dim) {
  int equals = TRUE;

  // compare memory buffers
  for (unsigned int y=0; y < y_dim; y++) {
    if (memcmp(buffer_a[y], buffer_b[y], x_dim * sizeof(double)) != 0) {
        equals = FALSE;
    }
  }

  return equals;
}

// compare vias vectors
int compareBiasVectors(double *buffer_a, double *buffer_b, unsigned int dim) {
  int equals = TRUE;

  // compare memory buffers
  if (memcmp(buffer_a, buffer_b, dim * sizeof(double)) != 0) {
      equals = FALSE;
  }

  return equals;
}

// dump weights to consosle
void displayWeights(double **buffer, unsigned int x_dim, unsigned int y_dim) {
  // loopover
  for (unsigned int y=0; y < y_dim; y++) {
    printf("| %d |", y);
    for (unsigned int x=0; x < x_dim; x++) {
      printf(" %e |", indexWeightsMatrix(buffer, x, y));
    }
    printf("\n");
  }
}

// get value @ position (x,y) in the matrix
double indexWeightsMatrix(double **buffer, unsigned int x_pos, unsigned int y_pos) {
  return (double)((buffer[y_pos])[x_pos]);
}

