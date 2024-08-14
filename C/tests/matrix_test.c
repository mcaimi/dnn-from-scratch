#include "matrix.h"
#include "matmult.h"

#define ROWS 3
#define COLS 5

int main(int argc, char **argv) {
  // allocation
  printf("Random Matrix:\n");
  double **rMatrix = randomMatrix(ROWS, COLS);
  displayWeights(rMatrix, ROWS, COLS);

  printf("Constant Matrix:\n");
  double **cMatrix = constantMatrix(ROWS, COLS, 5.5f);
  displayWeights(cMatrix, ROWS, COLS);

  printf("Zero Matrix:\n");
  double **zMatrix = zeroMatrix(ROWS, COLS);
  displayWeights(zMatrix, ROWS, COLS);

  freeMatrix(rMatrix, ROWS);
  freeMatrix(cMatrix, ROWS);
  freeMatrix(zMatrix, ROWS);

  // multiplication
  double *in_vector = randomVector(ROWS);
  double **in_matrix = randomMatrix(ROWS, COLS);
  printf("INPUT VECTOR\n");
  displayWeights(&in_vector, 1, ROWS);
  printf("INPUT MATRIX\n");
  displayWeights(in_matrix, ROWS, COLS);
  printf("OUTPUT VECTOR\n");
  double *out_vector = vec2Mat(in_vector, in_matrix, ROWS, COLS);
  displayWeights(&out_vector, 1, COLS);

  freeMatrix(in_matrix, ROWS);
  free(in_vector);
  free(out_vector);
}
