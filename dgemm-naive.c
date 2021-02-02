
#include <immintrin.h>

const char *dgemm_desc = "SIMD dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
//void square_dgemm(int n, double *A, double *B, double *C) {
//    // For each row i of A
//    for (int i = 0; i < n; i += 4) {
//        // For each column j of B
//        for (int j = 0; j < n; j++) {
//            __m256d c0 = {0, 0, 0, 0};
////            double cij = C[i + j * n];
//            for (int k = 0; k < n; k++) {
//                c0 = _mm256_add_pd(
//                        c0,
//                        _mm256_mul_pd(
//                                _mm256_loadu_pd(A + i + k * n),
////                                _mm256_loadu_pd(&A[i+k*n]),
////                                _mm256_broadcastu_sd(B+k+j*n)));
////                                _mm256_broadcast_sd(&B[k+j*n])));
////                                _mm256_loadu_pd(&B[k+j*n])));
//                                _mm256_loadu_pd(B + k + j * n)));
////                                _mm256_loadu_pd(&B[k+j*n])));
////                cij += A[i + k * n] * B[k + j * n];
//            }
//            _mm256_storeu_pd(C + i + j * n, c0);
////            _mm256_storeu_pd(&C[i+j*n], c0);
////            C[i + j * n] = cij;
//        }
//    }
//}


//static void mmul_saxpy_avx(const int n, const float *left, const float *right, float *result) {
//    int in = 0;
//    for (int i = 0; i < n; ++i) {
//        int kn = 0;
//        for (int k = 0; k < n; ++k) {
//            __m256 aik = _mm256_set1_ps(left[in + k]);
//            int j = 0;
//            for (; j < n; j += 8) {
//                _mm256_store_ps(result + in + j,
//                                _mm256_fmadd_ps(aik, _mm256_load_ps(right + kn + j), _mm256_load_ps(result + in + j)));
//            }
//            for (; j < n; ++j) {
//                result[in + j] += left[in + k] * right[kn + j];
//            }
//            kn += n;
//        }
//        in += n;
//    }
//}

void square_dgemm(int n, double *A, double *B, double *C) {
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < n; j++) {
            __m512d bkj = _mm512_set1_pd(B[k + j * n]);
            int i = 0;
            for (; i < n - 8; i += 8) {
                _mm512_storeu_pd(&C[i + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + k * n]), bkj,
                                                                _mm512_loadu_pd(&C[i + j * n])));
            }            //remaining elements
            for (; i < n; i++) {
                C[i + j * n] += A[i + k * n] * B[k + j * n];
            }
        }
    }