#!/bin/zsh
# TODO: Fix above for windows...

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "'conda' is not installed. Please install it first from https://docs.anaconda.com/free/miniconda/miniconda-install/"
    exit
fi

if { conda env list | grep 'mushroom-harvest'; } >/dev/null 2>&1;
then
    echo "Conda environment 'mushroom-harvest' already exists. Skipping environment creation."
else
    # Create a new conda environment
    conda create -y -n mushroom-harvest python=3.8
fi
