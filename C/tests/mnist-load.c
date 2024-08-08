#include "common.h"
#include "mnist.h"

mnist_index *test_index;
mnist_data *test_data;

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("Syntax: %s <indexFileName> <dataFileName>\n", argv[0]);
    exit(-1);
  }

  // test loading labels
  test_index = mnistLoadIndex(argv[1]);
  if (!test_index) {
    printf("Error loading index file");
    exit(-1);
  }
  mnistFreeIndex(test_index);

  // test loading images
  test_data = mnistLoadData(argv[2]);
  if (!test_data) {
    printf("Error loading data file");
    exit(-1);
  }

  // get a sample
  double *sample = mnistIndexData(arc4random()%test_data->n_items, test_data, 0);
  displaySample(sample, test_data);

  free(sample);
  mnistFreeData(test_data);
}
