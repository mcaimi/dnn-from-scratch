#include "matmult.h"

// product between vector v and matrix A
// v is a row vector (or transpose c(T) of column vector c) of length 1xK
// A is an KxP matrix
// => v*A = j, being j a vector of length 1xP
double *vec2Mat(double *input_vector, double **input_matrix, unsigned int mat_rows, unsigned int mat_cols) {
  // allocate output vector
  double *output_vector = constantVector(mat_cols, 0.0);

  // loop over output columns
  for (unsigned int c=0; c<mat_cols; c++) {
    double accumulator = 0.0;
    for (unsigned int r=0; r<mat_rows; r++) {
      accumulator += input_vector[r] * indexWeightsMatrix(input_matrix, r, c);
    }
    output_vector[c] = accumulator;
  }

  // return data
  return output_vector;
}
