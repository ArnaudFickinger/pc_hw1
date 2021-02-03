
#include <immintrin.h>

const char *dgemm_desc = "SIMD dgemm.";

void square_dgemm(int n, double *A, double *B, double *C) {
    for (int k = 0; k < n; k++) {
        for (int j = 0; j < n; j++) {
            __m512d bkj = _mm512_set1_pd(B[k + j * n]);
            int i = 0;
            for (; i < n - 64; i += 64) {
                _mm512_storeu_pd(&C[i + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + k * n]), bkj,
                                                                _mm512_loadu_pd(&C[i + j * n])));
                _mm512_storeu_pd(&C[i + 8 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 8 + k * n]), bkj,
                                                                    _mm512_loadu_pd(&C[i + 8 + j * n])));
                _mm512_storeu_pd(&C[i + 16 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 16 + k * n]), bkj,
                                                                     _mm512_loadu_pd(&C[i + 16 + j * n])));
                _mm512_storeu_pd(&C[i + 24 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 24 + k * n]), bkj,
                                                                     _mm512_loadu_pd(&C[i + 24 + j * n])));
                _mm512_storeu_pd(&C[i + 32 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 32 + k * n]), bkj,
                                                                     _mm512_loadu_pd(&C[i + 32 + j * n])));
                _mm512_storeu_pd(&C[i + 40 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 40 + k * n]), bkj,
                                                                     _mm512_loadu_pd(&C[i + 40 + j * n])));
                _mm512_storeu_pd(&C[i + 48 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 48 + k * n]), bkj,
                                                                     _mm512_loadu_pd(&C[i + 48 + j * n])));
                _mm512_storeu_pd(&C[i + 56 + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + 56 + k * n]), bkj,
                                                                     _mm512_loadu_pd(&C[i + 56 + j * n])));
                //remaining elements
            }
            for (; i < n - 8; i += 8) {
                _mm512_storeu_pd(&C[i + j * n], _mm512_fmadd_pd(_mm512_loadu_pd(&A[i + k * n]), bkj,
                                                                _mm512_loadu_pd(&C[i + j * n])));
            }            //remaining elements
                for (; i < n; i++) {
                    C[i + j * n] += A[i + k * n] * B[k + j * n];
                }
            }
        }
    }