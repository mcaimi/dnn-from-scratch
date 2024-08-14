#include "matmult.h"

// product between vector v and matrix A
// v is a row vector (or transpose c(T) of column vector c) of length K
// A is an KxP matrix
// => v*A = j, being j a row vector of size 1xP
double *vec2Mat(double *input_vector, double **input_matrix, unsigned int mat_rows, unsigned int mat_cols) {
  // allocate output vector
  double *output_vector = constantVector(mat_cols, 0.0f);

  // loop over output columns
  for (unsigned int p=0; p<mat_cols; p++) {
    double accumulator = 0.0f;
    for (unsigned int k=0; k<mat_rows; k++) {
      accumulator += input_vector[k] * indexWeightsMatrix(input_matrix, k, p);
    }
    output_vector[p] = accumulator;
  }

  // return data
  return output_vector;
}
