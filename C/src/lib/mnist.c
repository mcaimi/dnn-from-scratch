#include "mnist.h"
#include "common.h"
#include <time.h>

// ascii shades
const char *shades = " .'`^\",:;Il!i><~+_-?][}{1)(\\|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@";

// swap 32 bit integer bytes
uint32_t byteSwap(uint32_t value) {
  uint32_t temp = 0x00000000;

  temp |= (((uint8_t)(value&0xFF)) << 24)|(((uint8_t)(value>>8)&0xFF) << 16)|(((uint8_t)(value>>16)&0xFF) << 8)|((uint8_t)(value>>24)&0xFF);

  return temp;
}

// load labels
mnist_index *mnistLoadIndex(char *filename) {
  FILE *index_descriptor = NULL;
  mnist_index *temp = (mnist_index *)malloc(sizeof(struct __mnist_index_t));
  if (!temp) {
    printf("Cannot allocate memory for the index structure");
    return NULL;
  }

  // open file for reading
  index_descriptor = fopen(filename, "r");
  if (!index_descriptor) {
    printf("Failed to open %s\n", filename);
    mnistFreeIndex(temp);
    return NULL;
  }

  // read metadata
  fread(&temp->magic, sizeof(uint32_t), 1, index_descriptor);
  fread(&temp->n_items, sizeof(uint32_t), 1, index_descriptor);
  if (IS_LE) {
    printf("Little Endian System: Swapping Bytes...\n");
    temp->magic = byteSwap(temp->magic);
    temp->n_items = byteSwap(temp->n_items);
  }

  // display metadata
  printf("Loaded Index file %s: Magic Bytes: 0x%X, Indexed Items: %u\n", filename, temp->magic, temp->n_items);

  // load labels
  temp->labels = (uint8_t *)malloc(temp->n_items * sizeof(uint8_t));
  if (!temp->labels) {
    printf("Error while allocating memory for labels vector\n");
    mnistFreeIndex(temp);
    fclose(index_descriptor);
    return NULL;
  }
  for (unsigned int i=0; i < temp->n_items; i++) {
    fread((temp->labels + i), sizeof(uint8_t), 1, index_descriptor);
  }

  // close stream
  fclose(index_descriptor);

  // return data
  return temp;
}

// load data
mnist_data *mnistLoadData(char *filename) {
  FILE *data_descriptor = NULL;
  mnist_data *temp = (mnist_data *)malloc(sizeof(struct __mnist_data_t));
  if (!temp) {
    printf("Cannot allocate memory for the data structure");
    return NULL;
  }

  // open file for reading
  data_descriptor = fopen(filename, "r");
  if (!data_descriptor) {
    printf("Failed to open %s\n", filename);
    return NULL;
  }

  // read metadata
  fread(&temp->magic, sizeof(uint32_t), 1, data_descriptor);
  fread(&temp->n_items, sizeof(uint32_t), 1, data_descriptor);
  fread(&temp->n_rows, sizeof(uint32_t), 1, data_descriptor);
  fread(&temp->n_cols, sizeof(uint32_t), 1, data_descriptor);
  if (IS_LE) {
    printf("Little Endian System: Swapping Bytes...\n");
    temp->magic = byteSwap(temp->magic);
    temp->n_items = byteSwap(temp->n_items);
    temp->n_rows = byteSwap(temp->n_rows);
    temp->n_cols = byteSwap(temp->n_cols);
  }

  // display metadata
  printf("Loaded Index file %s: Magic Bytes: 0x%X, Data Items: %u\n", filename, temp->magic, temp->n_items);
  printf("\tNumber of Rows/Cols: %u/%u, Bytes per Item: %u\n", temp->n_rows, temp->n_cols, (temp->n_cols*temp->n_rows));

  // load data
  temp->data = (uint8_t **)malloc(temp->n_items * sizeof(uint8_t *));
  if (!temp->data) {
    printf("Error while allocating memory for labels vector\n");
    mnistFreeData(temp);
    fclose(data_descriptor);
    return NULL;
  }
  for (unsigned int i=0; i < temp->n_items; i++) {
    temp->data[i] = (uint8_t *)malloc(sizeof(uint8_t) * (temp->n_rows*temp->n_cols));
    fread(temp->data[i], sizeof(uint8_t), (temp->n_rows * temp->n_cols), data_descriptor);
  }

  // close stream
  fclose(data_descriptor);

  // return data
  return temp;
}

// free index resources
void mnistFreeIndex(mnist_index *item) {
  if (item->labels) {
    free(item->labels);
  }

  free(item);
}

// free data resources
void mnistFreeData(mnist_data *item) {
  if (item->data) {
    for (unsigned int i=(item->n_items - 1); i > 0; i--) {
      if (item->data[i]) {
        free(item->data[i]);
      }
    }
    free(item->data);
  }

  free(item);
}

// get an image from the training dataset
double *mnistIndexData(unsigned int idx, mnist_data *dataset, int normalize) {
  double *datapoint = NULL;
  uint8_t sample;

  // allocate memory...
  unsigned int num_pixels = dataset->n_rows * dataset->n_cols;
  datapoint = constantVector(num_pixels, 0.0f);
  if (!datapoint) return NULL;

  // index the dataset and return a datapoint
  for (unsigned int pixel=0; pixel < num_pixels; pixel++) {
    // cast uint8 to double
    sample = (dataset->data[idx])[pixel];
    if (normalize) {
      datapoint[pixel] = (double)(sample)/0xFF;
    } else {
      datapoint[pixel] = (double)(sample);
    }
  }

  // return it
  return datapoint;
}

// display sample
void displaySample(double *sample, mnist_data *dataset, int normalize) {
  for (unsigned int r=0; r<dataset->n_rows; r++) {
    for (unsigned int c=0; c<dataset->n_cols; c++) {
      if (normalize) {
        printf("%c", shades[(uint8_t)(sample[r*dataset->n_cols + c] * 0xFF) % (sizeof(shades)/sizeof(shades[0]))]);
      } else {
        printf("%c", shades[(uint8_t)(sample[r*dataset->n_cols + c]) % (sizeof(shades)/sizeof(shades[0]))]);
      }
    }
    printf("\n");
  }
}

