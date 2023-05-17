#include "OperatorApplier.hpp"
#include "matrix_vector_ops.hpp"
#include "cuda_memory_utils.hpp"
#include "iostream"

/**
 * @brief Constructor
 * 
 * @param m, n, q_nnz, q_i, q_j, q_v, a_nnz, a_i, a_j, a_v - CSR information for matrix Q and A
 * handle - cusparse context handle
 * 
 * @pre Q is nxn SPD, A is mxn (m > n), H is nxn diagonal
 * 
 * @post matrices are loaded, operator is automatically set to quadratic
*/
OperatorApplier::OperatorApplier(int m, 
    int n, 
    int q_nnz, 
    int* q_i, 
    int* q_j, 
    double* q_v, 
    int a_nnz, 
    int* a_i, 
    int* a_j, 
    double* a_v,
    cusparseHandle_t handle)
    : OperatorApplier(m, n, a_nnz, a_i, a_j, a_v, handle)
{
    load_Q_matrix(q_nnz, q_i, q_j, q_v);
}

/**
 * @brief Constructor that does not load Q matrix
 */
OperatorApplier::OperatorApplier(int m, 
    int n, 
    int a_nnz, 
    int* a_i, 
    int* a_j, 
    double* a_v,
    cusparseHandle_t handle)
    : n_{n}, 
    m_{m},
    a_nnz_{a_nnz}, 
    a_i_{a_i}, 
    a_j_{a_j}, 
    a_v_{a_v},
    handle_{handle}
{    
    void* buffer;
    allocateMatrixOnDevice(n_, a_nnz_, &a_t_i_, &a_t_j_, &a_t_v_);
    transposeMatrixOnDevice(handle_, m_, n_, a_nnz_, a_i_, a_j_, a_v_, a_t_i_, a_t_j_, a_t_v_, &buffer, false);
    deleteOnDevice(buffer);
    createCsrMat(&a_desc_, m_, n_, a_nnz_, a_i_, a_j_, a_v_);
    createCsrMat(&a_t_desc_, n_, m_, a_nnz, a_t_i_, a_t_j_, a_t_v_);

}

/**
 * @brief Destructor
*/
OperatorApplier::~OperatorApplier() 
{
    if (linear_allocated_) {
        deleteOnDevice(buffer1);
        deleteOnDevice(buffer2);
    }

    if (quadratic_allocated_) {
        deleteOnDevice(buffer3);
    }
}

/**
 * @brief loads CSR information for H matrix which is a diagonal matrix
 *
 * @param q_nnz - number nonzers in matrix Q
 * q_i, q_j, q_v - CSR rows, cols, values
 * 
 * @pre loaded Q matrix must have matching nxn dimensions with same nonzero structure
 * 
 * @post Q matrix is loaded, operator is automatically set to quadratic
 */
void OperatorApplier::load_Q_matrix(int q_nnz, int* q_i, int* q_j, double* q_v)
{
    q_nnz_= q_nnz; 
    q_i_ = q_i; 
    q_j_ = q_j;
    q_v_ = q_v;
    createCsrMat(&q_desc_, n_, n_, q_nnz_, q_i_, q_j_, q_v_);
    q_loaded_ = true;
    set_quadratic(true);
}

/**
 * @brief loads CSR information for H matrix which is a diagonal matrix
 *
 * @param h_v - vector representing the diagonal of H
 * 
 * @pre h_v has size m
 * 
 * @post Diagonal for H matrix is loaded
 */
void OperatorApplier::load_H_matrix(double* h_v)
{
    h_v_ = h_v;
}

/**
 * @brief sets scalar that multiplies Q
 * 
 * @param t - scalar
 * 
 * @pre t is some double value
 * 
 * @post t_ value is set to t
*/
void OperatorApplier::set_Q_scalar(double t) {
    t_ = t;
}

/**
 * @brief toggle if operator is quadratic or not
 * 
 * @param quadratic - boolean indicating whether Q matrix is used or not
 * 
 * @pre quadric is a boolean value
 * 
 * @post quadratic_ is set to quadratic boolean, can be true only if a Q matrix is already loaded
*/
void OperatorApplier::set_quadratic(bool quadratic) {
    quadratic_ = quadratic;
    if (!q_loaded_) {
        quadratic_ = false;
    }
}

/**
 * @brief applies the operatotr (tQ + A'HA) to v where Q is nxn, 
 *        A is mxn (m > n), H is diagonal mxm
 *
 * @param v - the vector to which the loaded operator is applied to
 * out - the result of applying the operator is copied into this vector
 * 
 * @pre v and out have size n
 * 
 * @post out holds the result of applying the operator to v
 */
void OperatorApplier::apply(double* v, double* out)
{   
    //TODO: consider not doing inplace of Q
    cusparseDnVecDescr_t v_desc;
    cusparseDnVecDescr_t out_desc;    
    createDnVec(&v_desc, n_, v);
    createDnVec(&out_desc, n_, out);

    if (!linear_allocated_) {
        allocateVectorOnDevice(m_, &w_);
        createDnVec(&w_desc_, m_, w_);
    }

    SpMV_product_reuse(handle_, ONE, a_desc_, v_desc, ZERO, w_desc_, &buffer1, linear_allocated_); //Av
    fun_vec_scale(m_, w_, h_v_); //HAv

    SpMV_product_reuse(handle_, ONE, a_t_desc_, w_desc_, ZERO, out_desc, &buffer2, linear_allocated_); //AtHAv

    if (quadratic_) {
        if (!quadratic_allocated_) {
            allocateVectorOnDevice(n_, &r_);
            createDnVec(&r_desc_, n_, r_);
        }
        SpMV_product_reuse(handle_, ONE, q_desc_, v_desc, ZERO, r_desc_, &buffer3, quadratic_allocated_); //Qv
        displayDeviceVector(r_, n_, 0, n_);
        fun_add_vecs(n_, out, t_, r_); //tQv + A'HAv
        quadratic_allocated_ = true;
    }

    linear_allocated_ = true;
}