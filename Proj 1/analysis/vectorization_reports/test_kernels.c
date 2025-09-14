
#include <stdlib.h>

typedef float f32;
typedef double f64;

// AXPY kernels
void axpy_scalar_f32(f32 a, f32* x, f32* y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void axpy_vectorized_f32(f32 a, f32* x, f32* y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void axpy_scalar_f64(f64 a, f64* x, f64* y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

void axpy_vectorized_f64(f64 a, f64* x, f64* y, size_t n) {
    for (size_t i = 0; i < n; i++) {
        y[i] = a * x[i] + y[i];
    }
}

// Dot product kernels
void dot_product_scalar_f32(f32* x, f32* y, size_t n, f32* result) {
    f32 sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    *result = sum;
}

void dot_product_vectorized_f32(f32* x, f32* y, size_t n, f32* result) {
    f32 sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += x[i] * y[i];
    }
    *result = sum;
}

// Test function to ensure code is used
void test_kernels() {
    f32 x[16], y[16], result;
    f64 dx[16], dy[16];
    
    // Test all kernels to ensure they're not optimized away
    axpy_scalar_f32(2.0f, x, y, 16);
    axpy_vectorized_f32(2.0f, x, y, 16);
    axpy_scalar_f64(2.0, dx, dy, 16);
    axpy_vectorized_f64(2.0, dx, dy, 16);
    dot_product_scalar_f32(x, y, 16, &result);
    dot_product_vectorized_f32(x, y, 16, &result);
}

int main() {
    test_kernels();
    return 0;
}
