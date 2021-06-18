#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include <cuda.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
    
#define DEPTH 5 // no of layers
#define BREADTH 1000 // nodes per layer
#define RATE 0.0001 // learning rate

#define NDIM(a,b) ((a)*BREADTH+(b)) 
#define WDIM(a,b,c) ((a)*BREADTH*BREADTH+(b)*BREADTH+(c))

float *W, *R; // [DEPTH-1][BREADTH][BREADTH] W_kij : weight of edge b/w jth and ith node from kth layer and k+1th layer resp. 
float *N; // [DEPTH][BREADTH] N_ki : jth node in kth layer
float *Y, *pY; // [BREADTH] expected and predicted output vectors
float *U, *V; // [BREADTH] vectors for backpropagation chain-rule

float Yh[BREADTH], Dh[BREADTH]; // host arrays: expected and predicted output
int count = 0;

#define f(x) (tanhf((x))) // activation function
#define df(y) (1 - (y) * (y)) // derivative of activation function in y terms


__global__ void predict_N( float *N) { // feedforward pass algorithm
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int i;
     // erase all nodes except first layer
    for ( i = tid + BREADTH ; i < BREADTH * DEPTH ; i += stride )
        N[i] = 0;
}


__global__ void predict_WN( float *N, float *W, float *R, int k) { // feedforward pass algorithm
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int h;

    // matrix-vector multiplication W*N b/w k-1th and kth layer
    int wIndex = k * BREADTH * BREADTH; 
    int nIndex = k * BREADTH; 
    for ( h = tid ; h < BREADTH * BREADTH ; h += stride ) {
        int j = h % BREADTH;
        R[h] = W[wIndex + h] * N[nIndex + j];
    }
}

__global__ void predict_Ri( float *R, int n, int j) { // feedforward pass algorithm
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int i;
    for ( i = tid % BREADTH ; i < BREADTH ; i += stride ) {

        int itid = tid / BREADTH;
        int istride = stride / BREADTH;
        istride += (istride == 0);

        int rIndex = i * BREADTH;
        
        for ( int h = 2 * j * itid; h < j * (n - 1); h += 2 * j * istride)
            R[rIndex + h] += R[rIndex + h + j];
    }
}

__global__ void predict_Nk( float *N, float *R, int k) { // feedforward pass algorithm
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int i;    
    for ( i = tid; i < BREADTH; i += stride )
        N[NDIM(k,i)] = f( R[NDIM(i,0)]);
}

__global__ void predict_Yp( float *N, float *pY) { // feedforward pass algorithm
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int i;    
    for ( i = tid; i < BREADTH; i += stride )
        pY[i] = N[NDIM(DEPTH-1,i)];
}


__global__ void train_V( float *N, float *Y, float *V) {    

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    int q = DEPTH - 2;

    // caching chain-rule calculation in vector to aid in backpropagation
    for ( int i = tid; i < BREADTH; i += stride ) {
        V[i] = (N[NDIM(q+1,i)] - Y[i]) * df( N[NDIM(q+1,i)]);
    }
}

__global__ void train_Wf( float *N, float *W, float *V) {    

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    int q = DEPTH - 2, h;

    // update weights of outer layer
    for ( h = tid; h < BREADTH * BREADTH; h += stride ) {
        int i = h / BREADTH;
        int j = h % BREADTH;
        W[WDIM(q,i,j)] -= (float)RATE * V[i] * N[NDIM(q,j)];
    }
}

__global__ void train_WN( float *N, float *W, float *A, float *R, int k) { // feedforward pass algorithm
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int h;

    // matrix-vector multiplication W*N b/w k-1th and kth layer
    int wIndex = k * BREADTH * BREADTH; 
    int nIndex = k * BREADTH; 
    for ( h = tid ; h < BREADTH * BREADTH ; h += stride ) {
        int j = h % BREADTH;
        int i = h / BREADTH;
        R[NDIM(j,i)] = A[i] * W[wIndex + h] * df( N[nIndex + j]);
    }
}

__global__ void train_B( float *R, float *B) { // feedforward pass algorithm
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int i;    
    for ( i = tid; i < BREADTH; i += stride )
        B[i] = R[NDIM(i,0)];
}

__global__ void train_Wh( float *N, float *W, float *A, int k) { // feedforward pass algorithm
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int h;

    // matrix-vector multiplication W*N b/w k-1th and kth layer
    int wIndex = k * BREADTH * BREADTH; 
    int nIndex = k * BREADTH; 
    for ( h = tid ; h < BREADTH * BREADTH ; h += stride ) {
        int j = h % BREADTH;
        int i = h / BREADTH;
        W[wIndex + h] -= (float)RATE * A[i] * N[nIndex + j];
    }
}


void train() {

    train_V<<<128, 32>>>( N, Y, V);
    cudaDeviceSynchronize();

    train_Wf<<<128, 32>>>( N, W, V);
    cudaDeviceSynchronize();
    
    int q = DEPTH - 2;

    float *A = &V[0], *B = &U[0];
    float *temp;

    // updating weights of hidden layers (backpropagation)
    for ( int k = q - 1; k >= 0; k-- ) {      

        // chain-rule expansion by multiplying with next layer's weight matrix and 
        // derivative of activation function on the same layer's nodes
        train_WN<<<128, 32>>>( N, W, A, R, k+1);
        cudaDeviceSynchronize();

        int j = 1;
        int n = BREADTH;
        while ( n > 1) {
            predict_Ri<<<128, 32>>>( R, n, j);
            cudaDeviceSynchronize();

            j *= 2;
            n = (n+1)/2;
        }

        train_B<<<128, 32>>>( R, B);
        cudaDeviceSynchronize();

        temp = B;        B = A;        A = temp;

        // adjust weights by gradient descent
        train_Wh<<<128, 32>>>( N, W, A, k);
        cudaDeviceSynchronize();
    }
}


__global__ void normalize( float *W, float *N, float *Y) {    

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int i, w = (DEPTH-1) * BREADTH * BREADTH;

    for ( i = tid; i < w; i+= stride)
        W[i] = 1/ (float)BREADTH;

    for ( i = tid; i < BREADTH; i+= stride)
        N[i] = i/ (float)BREADTH;

    for ( i = tid; i < BREADTH; i+= stride)
        Y[i] = i/ (float)BREADTH;
}

void initialize() {
    int w = (DEPTH-1) * BREADTH * BREADTH;

    // device memory allocation of global arrays
    cudaMalloc( (void**)(&W), sizeof(float) * w);
    cudaMalloc( (void**)(&R), sizeof(float) * BREADTH * BREADTH);
    cudaMalloc( (void**)(&N), sizeof(float) * DEPTH * BREADTH);
    cudaMalloc( (void**)(&Y), sizeof(float) * BREADTH);
    cudaMalloc( (void**)(&pY), sizeof(float) * BREADTH);
    cudaMalloc( (void**)(&U), sizeof(float) * BREADTH);
    cudaMalloc( (void**)(&V), sizeof(float) * BREADTH);

    /*
    // random number generator
    curandGenerator_t gen;

    // Create pseudo-random number generator 
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    // Set seed 
    curandSetPseudoRandomGeneratorSeed(gen, time(0));

    // Generate floats on device
    curandGenerateUniform(gen, W, w);
    curandGenerateUniform(gen, N, BREADTH * DEPTH);
    curandGenerateUniform(gen, Y, BREADTH);

    curandDestroyGenerator(gen);
    */
    
    normalize<<<128, 32>>>(W, N, Y);
    cudaDeviceSynchronize();
}

void calculateError() {
    float err = 0, dY;

    // dY : difference b/w expected and predicted output, and
    // err : sum of squares of dY

    for ( int i = 0 ; i < BREADTH ; i++ ) {
        dY = (Dh[i] - Yh[i]);
        err += dY * dY;
    }
    printf("ITERATION : %d\tLOSS : %f\t", count, err);
}

void freeResources() {
    cudaFree(Y);
    cudaFree(N);
    cudaFree(W);
    cudaFree(pY);
    cudaFree(U);
    cudaFree(V);
    cudaFree(R);
}

void predict() {

    predict_N<<<128, 32>>>(N);
    cudaDeviceSynchronize();
    
    for ( int k = 1; k < DEPTH ; k++ ) {
        predict_WN<<<128, 32>>>( N, W, R, k-1);
        cudaDeviceSynchronize();

        int j = 1;
        int n = BREADTH;
        while ( n > 1) {
            predict_Ri<<<128, 32>>>( R, n, j);
            cudaDeviceSynchronize();

            j *= 2;
            n = (n+1)/2;
        }

        predict_Nk<<<128, 32>>>( N, R, k);
        cudaDeviceSynchronize();
    }
    predict_Yp<<<128, 32>>>( N, pY);
    cudaDeviceSynchronize();
}

int main() {

    initialize();

    for ( count = 0; count < 4; count++) {

        clock_t begin, end;
        begin = clock();

        predict();
        train();

        end = clock();
        float CpuTime = (float)(end - begin) / CLOCKS_PER_SEC;

        CUDA_CALL(cudaMemcpy(Yh, Y, BREADTH * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(Dh, pY, BREADTH * sizeof(float), cudaMemcpyDeviceToHost));

        calculateError();
        printf("TIME: %f\n", CpuTime);
    }

    freeResources();

    return 0;
}
