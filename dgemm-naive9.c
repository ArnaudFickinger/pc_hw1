
#include <immintrin.h>

const char *dgemm_desc = "SIMD dgemm.";

void square_dgemm(int n, double *right, double *left, double *result) {
    const int block_width = n >= 256 ? 512 : 256;
    const int block_height = n >= 512 ? 8 : n >= 256 ? 16 : 32;
    for (int column_offset = 0; column_offset < n; column_offset += block_width) {
        for (int row_offset = 0; row_offset < n; row_offset += block_height) {
            for (int i = 0; i < n; ++i) {
                int j = column_offset;
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
//                for (; j < n; j++) {
//                    for (int k = row_offset; k < row_offset + block_height && k < n; ++k) {
//                        result[j + i * n] += left[k + i * n] * right[j + k * n];
//                    }
                }
            }
        }
    }

}
