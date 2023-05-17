#include <iostream>
#include <iomanip>
#include "input_functions.hpp"
#include "OperatorApplier.hpp"
#include "cuda_memory_utils.hpp"
#include "MMatrix.hpp"

template <typename T>
void printVec(T* vec, int size) {
  std::cout<<"[";
  for (int i = 0; i < size; i++) {
    std::string add = i == size - 1 ? "" : " ";
    std::cout<<std::setprecision(20)<<vec[i]<<add;
  }
  std::cout<<"]"<<std::endl;
}

/**
  * @brief Driver file demonstrates use of Operator Applier S = tQ + A'HA
  *
  * @pre Q is nxn SPD, A is mxn (m > n), H is nxn diagonal
  * 
  */
int main(int argc, char *argv[]) 
{  
    if(argc != 7)
    {
        printf("Incorrect number of inputs. Exiting ...\n");
        return -1;
    }
    
    int n;
    int m;
    int q_nnz;
    int* q_i = nullptr;
    int* q_j = nullptr;
    double* q_v = nullptr;

    int a_nnz;
    int* a_i = nullptr;
    int* a_j = nullptr;
    double* a_v = nullptr;

    double* h_v = nullptr;
    double* d_h_v;

    double* v_in = nullptr;
    double* d_v_in = nullptr;
    double* result_vec = nullptr;
    double* result_d_vec = nullptr;
    double* expect_vec = nullptr;
    //***************************FILE READING**************************//
    char const* const q_file_name = argv[1];
    char const* const a_file_name = argv[2];
    char const* const h_file_name = argv[3];
    char const* const v_file_name = argv[4];
    char const* const expect_file_name = argv[5];
    double t = std::stod(argv[6]);

    MMatrix mat_q = MMatrix();
    MMatrix mat_a = MMatrix();

    read_mm_file_into_coo(q_file_name, mat_q, 2);
    printf("\n/******* Q Matrix size: %d x %d nnz: %d *******/\n\n", 
      mat_q.n_, 
      mat_q.m_, 
      mat_q.nnz_);

    read_mm_file_into_coo(a_file_name, mat_a, 2);
    printf("\n/******* A Matrix size: %d x %d nnz: %d *******/\n\n", 
      mat_a.n_, 
      mat_a.m_, 
      mat_a.nnz_);

    sym_coo_to_csr(mat_q);
    coo_to_csr(mat_a);

    m = mat_a.n_; //rows
    n = mat_a.m_; //cols
    q_nnz = mat_q.nnz_;
    a_nnz = mat_a.nnz_;

    h_v = new double[m];
    v_in = new double[n];
    expect_vec = new double[n];
    read_rhs(h_file_name, h_v); //edit read file to give size of vector
    read_rhs(v_file_name, v_in);
    read_rhs(expect_file_name, expect_vec);
    
    if (mat_q.n_ != mat_a.m_ || mat_q.n_ != mat_q.m_) {
      printf("Invalid matrix dimensions. Exiting ...\n");
      return -1;
    }
    printf("File reading completed ..........................\n");
    //**************************MEMORY COPYING*************************//
    cloneMatrixToDevice(&mat_q, &q_i, &q_j, &q_v);
    cloneMatrixToDevice(&mat_a, &a_i, &a_j, &a_v);
    cloneVectorToDevice(m, &h_v, &d_h_v);
    cloneVectorToDevice(n, &v_in, &d_v_in);
    allocateVectorOnDevice(n, &result_d_vec);
    displayDeviceVector(q_i, n, 0, n, "q_i");
    cusparseHandle_t handle = NULL;
    createSparseHandle(handle);
    //************************APPLYING OPERATOR************************//
    OperatorApplier* applier = new OperatorApplier(m, n, q_nnz, q_i, q_j, q_v, a_nnz, a_i, a_j, a_v, handle);

    applier->set_Q_scalar(t);
    applier->load_H_matrix(d_h_v);
    applier->apply(d_v_in, result_d_vec);
    //*************************TESTING OPERATOR************************//
    int fails = 0;
    
    result_vec = new double[n];
    copyVectorToHost(n, result_d_vec, result_vec);
    for (int i = 0; i < n; i++) {
      double val1 = result_vec[i];
      double val2 = expect_vec[i];
      double diff = val1 - val2;
      if (abs(diff) > 1e-6) {
        printf("Error at index %d\nRESULT: %f EXPECTED: %f DIFFERENCE: %f\n", i, val1, val2, val1 - val2);
        fails++;
      }
    }
    //**************************FREEING MEMORY*************************//
    deleteOnDevice(d_h_v);
    deleteOnDevice(d_v_in);
    deleteOnDevice(result_d_vec);
    deleteMatrixOnDevice(q_i, q_j, q_v);
    deleteMatrixOnDevice(a_i, a_j, a_v);

    delete[] h_v;
    delete[] v_in;
    delete[] result_vec;
    delete[] expect_vec;
    
    delete applier;

    return fails;
}