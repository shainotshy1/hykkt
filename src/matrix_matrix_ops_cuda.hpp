#pragma once

#include <cuda_runtime.h>

__global__ void q_sparse_product(int n, int m, int q_nnz, int* q_i, int* q_j, double* q_v, 
int a_nnz, int* a_i, int* a_j, double* a_v, double* h_v, double* out);