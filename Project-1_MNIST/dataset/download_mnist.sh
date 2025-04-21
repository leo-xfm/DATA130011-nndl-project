#!/bin/bash

echo "Start Downloading MNIST..."

wget --no-check-certificate https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget --no-check-certificate https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
wget --no-check-certificate https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
wget --no-check-certificate https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz

echo "Finish Download. Unzipping ..."

gunzip *.gz

echo "Finish Unzipping."