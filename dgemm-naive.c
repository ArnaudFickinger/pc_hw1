// SSE SIMD intrinsics
#include <xmmintrin.h>
// AVX SIMD intrinsics
#include <immintrin.h>

const char* dgemm_desc = "SIMD dgemm.";

/*
 * This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values.
 */
void square_dgemm(int n, double* A, double* B, double* C) {
    // For each row i of A
    for (int i = 0; i < n; i+=4) {
        // For each column j of B
        for (int j = 0; j < n; j++) {
            __m256d c0 = {0,0,0,0};
//            double cij = C[i + j * n];
            for (int k = 0; k < n; k++) {
                c0 = _mm256_add_pd(
                        c0,
                        _mm256_mul_pd(
                                _mm256_load_pd(A+i+k*n),
                                _mm256_broadcast_sd(B+k+j*n)));
//                cij += A[i + k * n] * B[k + j * n];
            }
            _mm256_store_pd(C+i+j*n, c0);
//            C[i + j * n] = cij;
        }
    }
}
