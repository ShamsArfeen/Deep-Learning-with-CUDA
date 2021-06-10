#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <cuda.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define DEPTH 3 // no of layers
#define BREADTH 20 // nodes per layer
#define RATE 0.05 // learning rate

#define NDIM(a,b) ((a)*BREADTH+(b)) 
#define WDIM(a,b,c) ((a)*BREADTH*BREADTH+(b)*BREADTH+(c))

float *W; // [DEPTH-1][BREADTH][BREADTH] W_kij : weight of edge b/w jth and ith node from kth layer and k+1th layer resp. 
float *N; // [DEPTH][BREADTH] N_ki : jth node in kth layer
float *Y, *pY; // [BREADTH] expected and predicted output vectors
float *U, *V; // [BREADTH] vectors for backpropagation chain-rule

float Yh[BREADTH], Dh[BREADTH]; // host arrays: expected and predicted output
int count = 0;

#define f(x) (tanhf((x))) // activation function
#define df(y) (1 - (y) * (y)) // derivative of activation function in y terms

__global__ void predict( float *N, float *W, float *pY) { // feedforward pass algorithm
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int i, j, k;
     // erase all nodes except first layer
    for ( k = 1; k < DEPTH ; k++ ) {
        for ( i = tid ; i < BREADTH ; i+=stride ) {
            N[NDIM(k,i)] = 0;
        }
    }
    for ( k = 1; k < DEPTH ; k++ ) {
        // matrix-vector multiplication W*N b/w k-1th and kth layer
        for ( i = 0 ; i < BREADTH ; i++ ) {
            for ( j = tid; j < BREADTH; j+=stride ) {
                N[NDIM(k,i)] += W[WDIM(k-1,i,j)] * N[NDIM(k-1,j)];
            }
        }
        // activation function on kth layer
        for ( i = tid ; i < BREADTH ; i+=stride ) {
            N[NDIM(k,i)] = f( N[NDIM(k,i)] );
        }
    }
    for ( i = tid; i < BREADTH; i+=stride ) {
        pY[i] = N[NDIM(DEPTH-1,i)];
    }
}

__global__ void train( float *N, float *W, float *Y, float *pY, float *U, float *V) {    

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    //float V[BREADTH], U[BREADTH];
    int q = DEPTH - 2;

    // caching chain-rule calculation in vector to aid in backpropagation
    for ( int i = tid; i < BREADTH; i+=stride ) {
        V[i] = (N[NDIM(q+1,i)] - Y[i]) * df( N[NDIM(q+1,i)]);
    }
    
    // update weights of outer layer
    for ( int i = tid; i < BREADTH; i+=stride ) {
        for ( int j = 0 ; j < BREADTH ; j++ ) {
            W[WDIM(q,i,j)] -= (float)RATE * V[i] * N[NDIM(q,j)];
        }
    }
    float *A = &V[0], *B = &U[0];
    float *temp;

    // updating weights of hidden layers (backpropagation)
    for ( int k = q - 1; k >= 0; k-- ) {      

        // chain-rule expansion by multiplying with next layer's weight matrix and 
        // derivative of activation function on the same layer's nodes
        for ( int i = tid; i < BREADTH; i+=stride ) {
            for ( int j = 0; j < BREADTH; j++ ) {
                B[j] += A[i] * W[WDIM(k+1,i,j)] * df( N[NDIM(k+1,j)]);
            }
        }   
        temp = B;        B = A;        A = temp;

        // adjust weights by gradient descent
        for ( int i = tid; i < BREADTH; i+=stride ) {
            for ( int j = 0; j < BREADTH; j++ ) {
                W[WDIM(k,i,j)] -= (float)RATE * A[i] * N[NDIM(k,j)];
            }
        }
    }
    for ( int i = tid; i < BREADTH; i+=stride ) {
        pY[i] = N[NDIM(DEPTH-1,i)];
    }
}

void initialize() {
    // device memory allocation
    cudaMalloc( (void**)(&W), sizeof(float) * (DEPTH-1) * BREADTH * BREADTH);
    cudaMalloc( (void**)(&N), sizeof(float) * DEPTH * BREADTH);
    cudaMalloc( (void**)(&Y), sizeof(float) * BREADTH);
    cudaMalloc( (void**)(&pY), sizeof(float) * BREADTH);
    cudaMalloc( (void**)(&U), sizeof(float) * BREADTH);
    cudaMalloc( (void**)(&V), sizeof(float) * BREADTH);

    // random number generator
    curandGenerator_t gen;

    // Create pseudo-random number generator 
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    // Set seed 
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    // Generate floats on device
    curandGenerateUniform(gen, W, (DEPTH-1) * BREADTH * BREADTH);
    curandGenerateUniform(gen, N, BREADTH * DEPTH);
    curandGenerateUniform(gen, Y, BREADTH);

    curandDestroyGenerator(gen);
}

void calculateError() {
    float err = 0, dY;

    // dY : difference b/w expected and predicted output, and
    // err : sum of squares of dY

    for ( int i = 0 ; i < BREADTH ; i++ ) {
        dY = (Dh[i] - Yh[i]);
        err += dY * dY;
    }
    printf("Iteration Count : %d  Loss Function : %f\n", count, err);
}

void freeResources() {
    cudaFree(Y);
    cudaFree(N);
    cudaFree(W);
    cudaFree(pY);
    cudaFree(U);
    cudaFree(V);
}

int main() {

    initialize();

    for ( count = 0; count < 1000; count++) {

        predict<<<1,20>>>(N, W, pY);
        CUDA_CALL(cudaDeviceSynchronize());

        train<<<1,20>>>(N, W, Y, pY, U, V);
        CUDA_CALL(cudaDeviceSynchronize());

        if (count % 100 == 0) {
            CUDA_CALL(cudaMemcpy(Yh, Y, BREADTH * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CALL(cudaMemcpy(Dh, pY, BREADTH * sizeof(float), cudaMemcpyDeviceToHost));
            calculateError();
        }
    }

    freeResources();

    return 0;
}
