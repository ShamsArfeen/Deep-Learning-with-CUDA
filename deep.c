#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define DEPTH 5 // no of layers
#define BREADTH 1000 // nodes per layer
#define RATE 0.0001 // learning rate

#define f(x) (tanh(x)) // activation function
#define df(y) (1 - (y) * (y)) // derivative of activation function in y terms

#define NDIM(a,b) ((a)*BREADTH+(b)) 
#define WDIM(a,b,c) ((a)*BREADTH*BREADTH+(b)*BREADTH+(c))

double *W; //[DEPTH-1][BREADTH][BREADTH]; // W_kij : weight of edge b/w jth and ith node from kth layer and k+1th layer resp. 
double *N; //[DEPTH][BREADTH]; // N_ki : jth node in kth layer
double *Y; //[BREADTH]; // expected output vector
double *V, *U;

int count = 0;

int predict() { 
    // feedforward pass algorithm
    int i, j, k;

     // erase all nodes except first layer
    for ( k = 1; k < DEPTH ; k++ ) {
        for ( i = 0 ; i < BREADTH ; i++ ) {
            N[NDIM(k,i)] = 0; //(i == 0); // one node hardcoded to 1 produces bias term for the next layer
        }
    }
    
    for ( k = 1; k < DEPTH ; k++ ) {
        
        // matrix-vector multiplication W*N b/w k-1th and kth layer
        for ( i = 0 ; i < BREADTH ; i++ ) {
            for ( j = 0; j < BREADTH; j++ ) {
                N[NDIM(k,i)] += W[WDIM(k-1,i,j)] * N[NDIM(k-1,j)];
            }
        }

        // activation function on kth layer
        for ( i = 0 ; i < BREADTH ; i++ )
            N[NDIM(k,i)] = f( N[NDIM(k,i)] );
    }
    return 0;
}

void calculateError() { 
    // prints the value of error function

    double err = 0, dY;
    int i, k = DEPTH - 1;

    // dY : difference b/w expected and predicted output, and
    // error : sum of squares of dY
    for ( i = 0 ; i < BREADTH ; i++ ) {
        dY = (N[NDIM(k,i)] - Y[i]);
        err += dY * dY;
    }
    printf("ITERATION : %d\tLOSS : %f\t", count, err);
}

int train() {
    int q = DEPTH - 2;

    // caching chain-rule calculation in vector to aid in backpropagation
    for ( int i = 0 ; i < BREADTH ; i++ ) {
        V[i] = (N[NDIM(q+1,i)] - Y[i]) * df( N[NDIM(q+1,i)]);
    }

    // update weights of outer layer
    for ( int i = 0 ; i < BREADTH ; i++ ) {
        for ( int j = 0 ; j < BREADTH ; j++ ) {
            W[WDIM(q,i,j)] -= RATE * V[i] * N[NDIM(q,j)];
        }
    }
    double *A = &V[0], *B = &U[0];
    double *temp;

    // updating weights of hidden layers (backpropagation)
    for ( int k = q - 1; k >= 0; k-- ) {     

        // chain-rule expansion by multiplying with next layer's weight matrix and 
        // derivative of activation function on the same layer's nodes
        for ( int i = 0; i < BREADTH; i++ ) {
            for ( int j = 0; j < BREADTH; j++ ) {
                B[j] += A[i] * W[WDIM(k+1,i,j)] * df( N[NDIM(k+1,j)]);
            }
        }   
        temp = B;        B = A;        A = temp;

        // adjust weights by gradient descent
        for ( int i = 0; i < BREADTH; i++ ) {
            for ( int j = 0; j < BREADTH; j++ ) {
                W[WDIM(k,i,j)] -= RATE * A[i] * N[NDIM(k,j)];
            }
        }
    }
    return 0;
}

void initialize() {

    int w = (DEPTH-1) * BREADTH * BREADTH;
    W = (double*) malloc(w * sizeof(double));
    N = (double*) malloc(BREADTH * DEPTH * sizeof(double));
    Y = (double*) malloc(BREADTH * sizeof(double));
    U = (double*) malloc(BREADTH * sizeof(double));
    V = (double*) malloc(BREADTH * sizeof(double));

    int i, j, k;
    srand(3);

     // randomly assigning weights
	for ( i = 0; i < DEPTH-1; i++ )
	for ( j = 0; j < BREADTH; j++ )
	for ( k = 0; k < BREADTH; k++ )
		W[WDIM(i,j,k)] = 1 / (double)BREADTH;

     // random input and output vectors
	for ( i = 0; i < BREADTH; i++ ) {
        N[NDIM(0,i)] = i / (double)BREADTH; //(rand() % 100) / (double)100;
        Y[i] = i / (double)BREADTH; //(rand() % 100) / (double)100;
    }
}

void freeResources() {
    free(N);
    free(W);
    free(Y);
    free(U);
    free(V);
}

int main() {

    initialize();

    for ( count = 0; count < 4; count++) {

        clock_t begin, end;
        begin = clock();
        
        predict();
        train();

        end = clock();
        double CpuTime = (double)(end - begin) / CLOCKS_PER_SEC;

        calculateError();
        printf("TIME: %f\n", CpuTime);
    }

    freeResources();

    return 0;
}
