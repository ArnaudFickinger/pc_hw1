
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
//            __m512d c0 = {0, 0, 0, 0};
////            double cij = C[i + j * n];
//            for (int k = 0; k < n; k++) {
//                c0 = _mm512_add_pd(
//                        c0,
//                        _mm512_mul_pd(
//                                _mm512_loadu_pd(A + i + k * n),
////                                _mm512_loadu_pd(&A[i+k*n]),
////                                _mm512_broadcastu_sd(B+k+j*n)));
////                                _mm512_broadcast_sd(&B[k+j*n])));
////                                _mm512_loadu_pd(&B[k+j*n])));
//                                _mm512_loadu_pd(B + k + j * n)));
////                                _mm512_loadu_pd(&B[k+j*n])));
////                cij += A[i + k * n] * B[k + j * n];
//            }
//            _mm512_storeu_pd(C + i + j * n, c0);
////            _mm512_storeu_pd(&C[i+j*n], c0);
////            C[i + j * n] = cij;
//        }
//    }
//}


//static void mmul_saxpy_avx(const int n, const float *left, const float *right, float *result) {
//    int in = 0;
//    for (int i = 0; i < n; ++i) {
//        int kn = 0;
//        for (int k = 0; k < n; ++k) {
//            __m512d aik = _mm512_set1_ps(left[in + k]);
//            int j = 0;
//            for (; j < n; j += 8) {
//                _mm512_storeu_pd(result + in + j,
//                                _mm512_fmadd_pd(aik, _mm512_loadu_pd(right + kn + j), _mm512_loadu_pd(result + in + j)));
//            }
//            for (; j < n; ++j) {
//                result[in + j] += left[in + k] * right[kn + j];
//            }
//            kn += n;
//        }
//        in += n;
//    }
//}

//void square_dgemm(int n, double *A, double *B, double *C) {
//    for (int k = 0; k < n; k++) {
//        for (int j = 0; j < n; j++) {
//            __m512d bkj = _mm512_set1_pd(B[k + j * n]);
//            int i = 0;
//            for (; i < n - 8; i += 8) {
//                _mm512_storeu_pd(&C[i + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + k * n]), bkj,
//                                                                _mm512_loadu_pd(&C[i + j * n])));
//            }            //remaining elements
////            for (; i < n; i++) {
////                C[i + j * n] += A[i + k * n] * B[k + j * n];
////            }
//        }
//    }
//}

//void square_dgemm(int n, double *A, double *B, double *C) {
//    for (int k = 0; k < n; k++) {
//        for (int j = 0; j < n; j++) {
//            __m512d bkj = _mm512_set1_pd(B[k + j * n]);
//            int i = 0;
//            for (; i < n - 64; i += 64) {
//                _mm512_storeu_pd(&C[i + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + k * n]), bkj,
//                                                                _mm512_loadu_pd(&C[i + j * n])));
//                _mm512_storeu_pd(&C[i + 8 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 8 + k * n]), bkj,
//                                                                    _mm512_loadu_pd(&C[i + 8 + j * n])));
//                _mm512_storeu_pd(&C[i + 16 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 16 + k * n]), bkj,
//                                                                     _mm512_loadu_pd(&C[i + 16 + j * n])));
//                _mm512_storeu_pd(&C[i + 24 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 24 + k * n]), bkj,
//                                                                     _mm512_loadu_pd(&C[i + 24 + j * n])));
//                _mm512_storeu_pd(&C[i + 32 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 32 + k * n]), bkj,
//                                                                     _mm512_loadu_pd(&C[i + 32 + j * n])));
//                _mm512_storeu_pd(&C[i + 40 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 40 + k * n]), bkj,
//                                                                     _mm512_loadu_pd(&C[i + 40 + j * n])));
//                _mm512_storeu_pd(&C[i + 48 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 48 + k * n]), bkj,
//                                                                     _mm512_loadu_pd(&C[i + 48 + j * n])));
//                _mm512_storeu_pd(&C[i + 56 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 56 + k * n]), bkj,
//                                                                     _mm512_loadu_pd(&C[i + 56 + j * n])));
//                //remaining elements
//            }
//            for (; i < n - 8; i += 8) {
//                _mm512_storeu_pd(&C[i + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + k * n]), bkj,
//                                                                _mm512_loadu_pd(&C[i + j * n])));
//            }            //remaining elements
//                for (; i < n; i++) {
//                    C[i + j * n] += A[i + k * n] * B[k + j * n];
//                }
//            }
//        }
//    }

void square_dgemm(int n, double *left, double *right, double *result) {
    const int block_width = n >= 256 ? 512 : 256;
    const int block_height = n >= 512 ? 8 : n >= 256 ? 16 : 32;
    for (int column_offset = 0; column_offset < n; column_offset += block_width) {
        for (int row_offset = 0; row_offset < n; row_offset += block_height) {
            for (int i = 0; i < n; ++i) {
                int j = column_offset
                for (; j < column_offset + block_width && j < n; j += 64) {
                    __m512d sum1 = _mm512_loadu_pd(result + i * n + j);
                    __m512d sum2 = _mm512_loadu_pd(result + i * n + j + 8);
                    __m512d sum3 = _mm512_loadu_pd(result + i * n + j + 16);
                    __m512d sum4 = _mm512_loadu_pd(result + i * n + j + 24);
                    __m512d sum5 = _mm512_loadu_pd(result + i * n + j + 32);
                    __m512d sum6 = _mm512_loadu_pd(result + i * n + j + 40);
                    __m512d sum7 = _mm512_loadu_pd(result + i * n + j + 48);
                    __m512d sum8 = _mm512_loadu_pd(result + i * n + j + 56);
                    for (int k = row_offset; k < row_offset + block_height && k < n; ++k) {
                        __m512d multiplier = _mm512_set1_pd(left[i * n + k]);
                        sum1 = _mm512_fmadd_pd(multiplier, _mm512_loadu_pd(right + k * n + j), sum1);
                        sum2 = _mm512_fmadd_pd(multiplier, _mm512_loadu_pd(right + k * n + j + 8), sum2);
                        sum3 = _mm512_fmadd_pd(multiplier, _mm512_loadu_pd(right + k * n + j + 16), sum3);
                        sum4 = _mm512_fmadd_pd(multiplier, _mm512_loadu_pd(right + k * n + j + 24), sum4);
                        sum5 = _mm512_fmadd_pd(multiplier, _mm512_loadu_pd(right + k * n + j + 32), sum5);
                        sum6 = _mm512_fmadd_pd(multiplier, _mm512_loadu_pd(right + k * n + j + 40), sum6);
                        sum7 = _mm512_fmadd_pd(multiplier, _mm512_loadu_pd(right + k * n + j + 48), sum7);
                        sum8 = _mm512_fmadd_pd(multiplier, _mm512_loadu_pd(right + k * n + j + 56), sum8);
                    }
                    _mm512_storeu_pd(result + i * n + j, sum1);
                    _mm512_storeu_pd(result + i * n + j + 8, sum2);
                    _mm512_storeu_pd(result + i * n + j + 16, sum3);
                    _mm512_storeu_pd(result + i * n + j + 24, sum4);
                    _mm512_storeu_pd(result + i * n + j + 32, sum5);
                    _mm512_storeu_pd(result + i * n + j + 40, sum6);
                    _mm512_storeu_pd(result + i * n + j + 48, sum7);
                    _mm512_storeu_pd(result + i * n + j + 56, sum8);
                }
                for (; j < n; j++) {
                    for (int k = row_offset; k < row_offset + block_height && k < n; ++k) {
                        result[j + i * n] += left[k + i * n] * right[j + k * n];
                    }
                }
            }
        }
    }

}
