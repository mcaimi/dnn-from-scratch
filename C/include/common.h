#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRUE 1
#define FALSE 0

// initialize random weights in memory
double **randomMatrix(unsigned int, unsigned int);

// initialize random weights in memory
double *randomVector(unsigned int);

// initialize a constant vector
double *constantVector(unsigned int, double);

// compare weights buffers
int compareWeightsBuffers(double **, double **, unsigned int, unsigned int);

// compare bias vectors
int compareBiasVectors(double *, double *, unsigned int);

// dump weights to console
void displayWeights(double **, unsigned int, unsigned int);

// get value @ position (x,y) in the matrix
double indexWeightsMatrix(double **, unsigned int, unsigned int);
