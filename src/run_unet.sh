#!/usr/bin/env bash

# Runs train.py and saves the console output to output/out
stdbuf -i0 -o0 -e0 python -u train_unet.py | tee output/out
