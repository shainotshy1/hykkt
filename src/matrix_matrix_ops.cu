#include "matrix_matrix_ops_cuda.hpp"

void fun_q_sparse_product(int n, 
int m, 
int q_nnz, 
int* q_i, 
int* q_j, 
double* q_v, 
int a_nnz, 
int* a_i, 
int* a_j, 
double* a_v, 
double* h_v, 
double* out) 
{
    int num_blocks;
    int block_size = 512;
    num_blocks = (n + block_size - 1) / block_size;
    q_sparse_product<<<num_blocks, block_size>>>(n, 
        m, 
        q_nnz, 
        q_i, 
        q_j, 
        q_v, 
        a_nnz, 
        a_i, 
        a_j, 
        a_v, 
        h_v, 
        out);
}

__global__ void q_sparse_product(int n, 
int m, 
int q_nnz, 
int* q_i, 
int* q_j, 
double* q_v, 
int a_nnz, 
int* a_i, 
int* a_j, 
double* a_v, 
double* h_v, 
double* out) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  double total = 0.0;
  int row_offset = 0;
}