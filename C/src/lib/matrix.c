#include "matrix.h"

// initialize random weights in memory
double **randomMatrix(unsigned int rows, unsigned int cols) {
  double **placeholder = NULL;

  // allocate space for the weight matrix
  // weights are arranged in a RowsxCols (YxX) matrix
  // values are double
  placeholder = (double **)malloc(rows * sizeof(double *));
  if (!placeholder) return NULL;

  // fill matrix with random numbers
  // each row is "cols" long (e.g. contains "cols" double values)
  for (unsigned int y=0; y < rows; y++) {
    placeholder[y] = randomVector(cols);
  }

  // return data
  return placeholder;
}

// initialize constant weights in memory
double **constantMatrix(unsigned int rows, unsigned int cols, double const_val) {
  double **placeholder = NULL;

  // allocate space for the weight matrix
  // weights are arranged in a RowsxCols (YxX) matrix
  // values are double
  placeholder = (double **)malloc(rows * sizeof(double *));
  if (!placeholder) return NULL;

  // fill matrix with random numbers
  // each row is "cols" long (e.g. contains "cols" double values)
  for (unsigned int y=0; y < rows; y++) {
    placeholder[y] = constantVector(cols, const_val);
  }

  // return data
  return placeholder;
}

// initialize zero matrix
double **zeroMatrix(unsigned int rows, unsigned int cols) {
  double **placeholder = NULL;

  // allocate a matrix full of zeros
  placeholder = constantMatrix(rows, cols, 0.0);

  // return data
  return placeholder;
}

// initialize random biases in memory
double *randomVector(unsigned int length) {
  double *placeholder = NULL;
  static int seeded = 0;

  // one-dimensional vector of double values
  placeholder = (double *)malloc(length * sizeof(double));
  if (!placeholder) return NULL;

  // seed random number generator only once
  if (!seeded) {
    srand(time(NULL));
    seeded = 1;
  }

  // fill vector with random numbers
  for (unsigned int i=0; i < length; i++) {
      placeholder[i] = ((double)arc4random()/(double)ARC4RANDOM_MAX);
  }

  return placeholder;
}

// initialize a constant vector
double *constantVector(unsigned int length, double const_val) {
  double *placeholder = NULL;

  // allocate buffer
  placeholder = (double *)malloc(length * sizeof(double));
  if (!placeholder) return NULL;

  // fill vector with random numbers
  for (unsigned int i=0; i < length; i++) {
      placeholder[i] = const_val;
  }

  return placeholder;

}

// copy vectors
void copyVectors(double *src, double *dst, unsigned int size) {
  // copy values from src to dst
  memcpy(dst, src, size*sizeof(double));
}

// compare weight buffers
int compareWeightsBuffers(double **buffer_a, double **buffer_b, unsigned int rows, unsigned int cols) {
  int equals = TRUE;

  // compare memory buffers
  for (unsigned int y=0; y < rows; y++) {
    if (memcmp(buffer_a[y], buffer_b[y], cols * sizeof(double)) != 0) {
        equals = FALSE;
    }
  }

  return equals;
}

// compare vias vectors
int compareBiasVectors(double *buffer_a, double *buffer_b, unsigned int size) {
  int equals = TRUE;

  // compare memory buffers
  if (memcmp(buffer_a, buffer_b, size * sizeof(double)) != 0) {
      equals = FALSE;
  }

  return equals;
}

// dump weights to consosle
void displayWeights(double **buffer, unsigned int rows, unsigned int cols) {
  // loopover
  for (unsigned int y=0; y < rows; y++) {
    printf("| %d |", y);
    for (unsigned int x=0; x < cols; x++) {
      printf(" %e |", indexWeightsMatrix(buffer, y, x));
    }
    printf("\n");
  }
}

// get value @ position (x,y) in the matrix
double indexWeightsMatrix(double **buffer, unsigned int row, unsigned int col) {
  return (double)((buffer[row])[col]);
}

// free an allocated matrix
void freeMatrix(double **buffer, unsigned int rows) {
  if (buffer) {
    for (unsigned int r=0; r<rows;r++) {
      if(buffer[r]) free(buffer[r]);
    }
    free(buffer);
  }
}
