# Deep-Learning-with-CUDA

A generic implementation of backpropagation algorithm in CUDA and C. Time of execution is upto within a constant factor of number of edges. Each layer has equal number of neurons including the hidden layers as well as the input and output layers. The number of intermediate layers can be controlled with compile-time constants namely, BREADTH and DEPTH. The constants represent the count of nodes per layer and the count of all layers respectively.

## Compilation CMDs
cmd for CUDA code:
`nvcc deep.cu -lcurand -o deepcuda`
cmd for C code:
`gcc -lm deep.c -o deepc`
