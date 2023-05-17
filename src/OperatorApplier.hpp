#pragma once

#include <cusparse.h>
#include <cublas.h>
#include "cusparse_utils.hpp"
#include "constants.hpp"

class OperatorApplier {
public:
    OperatorApplier(int m, 
    int n,
    int a_nnz, 
    int* a_i, 
    int* a_j, 
    double* a_v,
    cusparseHandle_t handle); 

    OperatorApplier(int m, 
    int n, 
    int q_nnz, 
    int* q_i, 
    int* q_j, 
    double* q_v, 
    int a_nnz, 
    int* a_i, 
    int* a_j, 
    double* a_v,
    cusparseHandle_t handle); 

    ~OperatorApplier();

    void load_Q_matrix(int q_nnz, int* q_i, int* q_j, double* q_v);
    void load_H_matrix(double* h_v);
    void set_Q_scalar(double t);
    void set_quadratic(bool quadratic);
    void apply(double* v, double* out);
private:

    cusparseHandle_t handle_; //cuspsarse handle context

    cusparseSpMatDescr_t q_desc_; //csr matrix structure for Q
    cusparseSpMatDescr_t a_desc_; //csr matrix structure for A
    cusparseSpMatDescr_t a_t_desc_; //csr matrix structure for A transpose
    
    cusparseDnVecDescr_t w_desc_; //vector structure for intermediate vector in matrix multiplication
    cusparseDnVecDescr_t r_desc_; //vector structure for intermediate vector in matrix multiplication

    bool linear_allocated_ = false; //boolean for if linear system buffers allocated
    bool quadratic_allocated_ = false; //boolean for if quadradic system buffers allocated
    bool quadratic_ = false; //boolean for if the operator is quadratic (i.e tQ is included)
    bool q_loaded_ = false; //boolean for if a Q matrix is loaded in the operator
    void* buffer1; //buffer used in Av product
    void* buffer2; //buffer used in AtHAv product
    void* buffer3; //buffer used in Qv product

    const int m_; //m dimension of mxn A matrix
    const int n_; //n dimension of mxn A matrix, nxn Q matrix, nxn H matrix, nx1 v vector

    double t_ = ONE; //scalar multiplying matrix Q

    int q_nnz_; //number of nonzeros in Q
    int* q_i_; //csr rows of Q
    int* q_j_; //csr cols of Q
    double* q_v_; //nonzero values of Q

    const int a_nnz_; //number of nonzeros in A
    int* a_i_; //csr rows of A
    int* a_j_; //csr cols of A
    double* a_v_; //nonzero values of A
    int* a_t_i_; //csr rows of A transpose
    int* a_t_j_; //csr cols of A transpose
    double* a_t_v_; //nonzero values of A transpose

    double* h_v_; //diagonal values of H

    double* w_; //intermediate vector used in matrix multiplaction, size m_
    double* r_; //intermediate vector used in matrix multiplaction, size n_
};