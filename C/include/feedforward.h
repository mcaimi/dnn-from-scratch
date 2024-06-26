#include <stdio.h>

// set input values into the neuron layer
void layerFeedIn(neuron *, double *);

// feed-forward matrix multiplication
double *matrixMultiplication(neuron *);

// gradient update
void gradientDescent(neuron *);

