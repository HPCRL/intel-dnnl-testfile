#!/bin/bash
module load gcc/9.1.0
g++ -std=c++11 -g  -fopenmp -I /home/li23/dnnl/mkl-dnn/install/include -L/home/li23/dnnl/mkl-dnn/install/lib64 myimpl.cpp -ldnnl