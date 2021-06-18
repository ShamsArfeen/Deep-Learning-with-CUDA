# Deep-Learning-with-CUDA

A generic implementation of backpropagation algorithm in CUDA and C, with time linear to the number of edges. Every layer has been assumed to have equal number of neurons - the hidden layers as well as the input and output layers. The size of layers and the number of intermediate layers can be controlled with compile-time constants namely, BREADTH and DEPTH. The constants represent the count of nodes per layer and the count of all layers, respectively. With increasing parallelism, BREADTH will have decreased effect on execution time. Activation function (tanh()) is also the same for all layers except that it can be changed to any other sigmoid function.

## Compilation Commands
For CUDA code:
`nvcc deep.cu -lcurand -o deepcuda`

For C code:
`gcc -lm deep.c -o deepc`

## Example Execution for Performance
1000 neurons, 5 layers and learning rate of 0.0001
![](http://github.com/ShamsArfeen/Deep-Learning-with-CUDA/blob/main/deeplearning.JPG?raw=true)

## Example Execution for Correctness
With a constant random seed, 30 neurons per layer, 3 layers and learning rate of 0.1
![](http://github.com/ShamsArfeen/Deep-Learning-with-CUDA/blob/main/deeplearning2.JPG?raw=true)
