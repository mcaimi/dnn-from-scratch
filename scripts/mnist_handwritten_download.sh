#!/bin/bash

# MNIST Handwritten DB download links
declare -a download_links=("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
                          "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
                          "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz"
                          "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz")

# get files
for link in ${download_links[@]}; do
  echo "Downloading ${link}..."
  IFS="/" read -ra LINK <<< "${link}"
  filename=${LINK[${#LINK[@]} - 1]}
  curl -o ${filename} ${link}
  echo "Decompressing ${filename}..."
  gunzip ${filename}
done


