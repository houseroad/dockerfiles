#!/bin/bash

python bench.py --model_name resnet18 --batch_sizes 1 2 8 16 --runs 10
