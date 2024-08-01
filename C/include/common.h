#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TRUE 1
#define FALSE 0

static union { uint8_t test[4]; uint32_t mem_repr; } endiannes_test = { {0xCA, 0x00, 0x00, 0xFE} };
#define IS_LE ((uint8_t)endiannes_test.mem_repr == 0xCA)
#define IS_BE ((uint8_t)endiannes_test.mem_repr == 0xFE)

// initialize random weights in memory
double **randomMatrix(unsigned int, unsigned int);

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
