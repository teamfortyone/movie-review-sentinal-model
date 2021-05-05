#!/bin/bash

# download dataset
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

# extract the dataset
tar -xf aclImdb_v1.tar.gz

# get pickled list of reviews from dataset

python3 utils/dataset.py
