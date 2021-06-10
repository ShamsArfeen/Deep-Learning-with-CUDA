#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define DEPTH 3 // no of layers
#define BREADTH 20 // nodes per layer
#define RATE 0.01 // learning rate

#define f(x) (tanh(x)) // activation function
#define df(y) (1 - (y) * (y)) // derivative of activation function in y terms

double W[DEPTH-1][BREADTH][BREADTH]; // W_kij : weight of edge b/w jth and ith node from kth layer and k+1th layer resp. 
double N[DEPTH][BREADTH]; // N_ki : jth node in kth layer
double Y[BREADTH]; // expected output vector

int count = 0;

int predict() { 
    // feedforward pass algorithm
    int i, j, k;

     // erase all nodes except first layer
    for ( k = 1; k < DEPTH ; k++ ) {
        for ( i = 0 ; i < BREADTH ; i++ ) {
            N[k][i] = 0;
        }
    }
    for ( k = 1; k < DEPTH ; k++ ) {
        // matrix-vector multiplication W*N b/w k-1th and kth layer
        for ( i = 0 ; i < BREADTH ; i++ ) {
            for ( j = 0; j < BREADTH; j++ ) {
                N[k][i] += W[k-1][i][j] * N[k-1][j];
            }
        }
        // activation function on kth layer
        for ( i = 0 ; i < BREADTH ; i++ )
            N[k][i] = f( N[k][i] );
    }
    return 0;
}

int calculateError() { 
    // prints the value of error function

    double err = 0, dY;
    int i, k = DEPTH - 1;

    // dY : difference b/w expected and predicted output, and
    // error : sum of squares of dY
    for ( i = 0 ; i < BREADTH ; i++ ) {
        dY = (N[k][i] - Y[i]);
        err += dY * dY;
    }
    printf("Iteration Count : %d  Loss Function : %f\n", count, err);
}

int train() {
    double V[BREADTH], U[BREADTH];
    int q = DEPTH - 2;

    predict();

    // caching chain-rule calculation in vector to aid in backpropagation
    for ( int i = 0 ; i < BREADTH ; i++ ) {
        V[i] = (N[q+1][i] - Y[i]) * df( N[q+1][i]);
    }

    // update weights of outer layer
    for ( int i = 0 ; i < BREADTH ; i++ ) {
        for ( int j = 0 ; j < BREADTH ; j++ ) {
            W[q][i][j] -= RATE * V[i] * N[q][j];
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
                B[j] += A[i] * W[k+1][i][j] * df( N[k+1][j]);
            }
        }   
        temp = B;        B = A;        A = temp;

        // adjust weights by gradient descent
        for ( int i = 0; i < BREADTH; i++ ) {
            for ( int j = 0; j < BREADTH; j++ ) {
                W[k][i][j] -= RATE * A[i] * N[k][j];
            }
        }
    }
    return 0;
}

int initialize() {
    int i, j, k;
    srand(time(0));

     // randomly assigning weights
	for ( i = 0; i < DEPTH-1; i++ )
	for ( j = 0; j < BREADTH; j++ )
	for ( k = 0; k < BREADTH; k++ )
		W[i][j][k] = (rand() % 100) / (double)1000;

     // random input and output vectors
	for ( i = 0; i < BREADTH; i++ ) {
        N[0][i] = (rand() % 100) / (double)100;
        Y[i] = (rand() % 100) / (double)100;
    }

    return 0;
}

int main() {

    initialize();

    for ( count = 0; count < 1000; count++) {
        train();
        
        if (count % 100 == 0) calculateError();
    }

    return 0;
}