#pragma once
#include <stdint.h>

struct __mnist_index_t {
  uint32_t magic;
  uint32_t n_items;
  uint8_t *labels;
};
typedef struct __mnist_index_t mnist_index;

struct __mnist_data_t {
  uint32_t magic;
  uint32_t n_items;
  uint32_t n_rows;
  uint32_t n_cols;
  uint8_t **data;
};
typedef struct __mnist_data_t mnist_data;

// magic bytes values
// first 2 bytes are always zero
// third byte meaning:
#define UNSIGNED_BYTE 0x08
#define SIGNED_BYTE 0x09
#define SHORT 0x0B
#define INT 0x0C
#define FLOAT 0x0D
#define DOUBLE 0x0E
// fourth byte encodes size
#define VECTOR 1
#define MATRIX 2

// swap 32 bit integer bytes MSB->LSB and vice-versa
uint32_t byteSwap(uint32_t);

// open dataset index
mnist_index *mnistLoadIndex(char *);

// open dataset
mnist_data *mnistLoadData(char *);

// free resources
void mnistFreeIndex(mnist_index *);
void mnistFreeData(mnist_data *);
