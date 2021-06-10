# Deep-Learning-with-CUDA

A generic implementation of backpropagation algorithm in CUDA and C, with time linear to the number of edges. Every layer has been assumed to have equal number of neurons - the hidden layers as well as the input and output layers. The number of intermediate layers and the size of layers can be controlled with compile-time constants namely, BREADTH and DEPTH. The constants represent the count of nodes per layer and the count of all layers, respectively. With increasing parallelism, BREADTH will have decreased effect on execution time.

## Compilation CMDs
cmd for CUDA code:
`nvcc deep.cu -lcurand -o deepcuda`

cmd for C code:
`gcc -lm deep.c -o deepc`
