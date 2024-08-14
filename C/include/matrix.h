#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <float.h>
#include <time.h>

#define TRUE 1
#define FALSE 0

#ifndef ARC4RANDOM_MAX
#define ARC4RANDOM_MAX 0x100000000
#endif

// initialize random weights in memory
double **randomMatrix(unsigned int, unsigned int);
double **constantMatrix(unsigned int, unsigned int, double);
double **zeroMatrix(unsigned int, unsigned int);

// initialize random weights in memory
double *randomVector(unsigned int);

// initialize a constant vector
double *constantVector(unsigned int, double);

// copy vector contents
void copyVectors(double *, double *, unsigned int);

// compare weights buffers
int compareWeightsBuffers(double **, double **, unsigned int, unsigned int);

// compare bias vectors
int compareBiasVectors(double *, double *, unsigned int);

// dump weights to console
void displayWeights(double **, unsigned int, unsigned int);

// get value @ position (x,y) in the matrix
double indexWeightsMatrix(double **, unsigned int, unsigned int);

// free matrix
void freeMatrix(double **, unsigned int);
