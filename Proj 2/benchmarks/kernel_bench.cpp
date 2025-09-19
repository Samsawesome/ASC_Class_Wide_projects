#include "../include/memory_utils.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <random>

// Lightweight kernel: SAXPY (Single-precision Alpha X Plus Y)
void saxpy(int n, float a, float* x, float* y) {
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}

// Test with different access patterns to vary cache miss rates
void test_cache_miss_impact(size_t working_set_size, int stride, bool random_access) {
    const int n = working_set_size / sizeof(float);
    float a = 2.0f;
    
    // Allocate aligned memory
    float* x = (float*)aligned_alloc(CACHE_LINE_SIZE, working_set_size);
    float* y = (float*)aligned_alloc(CACHE_LINE_SIZE, working_set_size);
    
    // Initialize arrays
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(n - i);
    }
    
    // Create access pattern indices if using random access
    std::vector<int> indices;
    if (random_access) {
        indices.resize(n);
        for (int i = 0; i < n; ++i) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), std::default_random_engine(42));
    }
    
    flush_cache(); // Clear cache before measurement
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (random_access) {
        // Random access pattern
        for (int i = 0; i < n; i += stride) {
            int idx = indices[i];
            y[idx] = a * x[idx] + y[idx];
        }
    } else {
        // Sequential access with stride
        for (int i = 0; i < n; i += stride) {
            y[i] = a * x[i] + y[i];
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_ns = std::chrono::duration<double, std::nano>(end - start).count();
    double operations = n / static_cast<double>(stride);
    double performance = operations / (time_ns / 1e9); // Operations per second
    
    std::cout << "Size: " << working_set_size << "B, Stride: " << stride 
              << ", Random: " << (random_access ? "Yes" : "No")
              << ", Time: " << time_ns << " ns, Perf: " << performance << " ops/s" << std::endl;
    
    aligned_free(x);
    aligned_free(y);
}

// Test TLB impact by varying page size and locality
void test_tlb_impact(size_t working_set_size, bool use_huge_pages) {
    const int n = working_set_size / sizeof(float);
    float a = 2.0f;
    
    // Allocate memory with different page options
    DWORD allocation_type = use_huge_pages ? MEM_LARGE_PAGES : 0;
    float* x = (float*)VirtualAlloc(NULL, working_set_size, MEM_COMMIT | MEM_RESERVE | allocation_type, PAGE_READWRITE);
    float* y = (float*)VirtualAlloc(NULL, working_set_size, MEM_COMMIT | MEM_RESERVE | allocation_type, PAGE_READWRITE);
    
    // Initialize arrays
    for (int i = 0; i < n; ++i) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(n - i);
    }
    
    flush_cache(); // Clear cache before measurement
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform SAXPY operation
    saxpy(n, a, x, y);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_ns = std::chrono::duration<double, std::nano>(end - start).count();
    double performance = n / (time_ns / 1e9); // Operations per second
    
    std::cout << "Size: " << working_set_size << "B, Huge Pages: " << (use_huge_pages ? "Yes" : "No")
              << ", Time: " << time_ns << " ns, Perf: " << performance << " ops/s" << std::endl;
    
    VirtualFree(x, 0, MEM_RELEASE);
    VirtualFree(y, 0, MEM_RELEASE);
}

int main() {
    std::cout << "=== Cache Miss Impact Tests ===" << std::endl;
    size_t sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216}; // 1KB to 16MB
    int strides[] = {1, 2, 4, 8, 16, 32, 64};
    
    for (size_t size : sizes) {
        for (int stride : strides) {
            test_cache_miss_impact(size, stride, false); // Sequential
            test_cache_miss_impact(size, stride, true);  // Random
        }
    }
    
    std::cout << "\n=== TLB Impact Tests ===" << std::endl;
    size_t tlb_sizes[] = {4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864}; // 4KB to 64MB
    
    for (size_t size : tlb_sizes) {
        test_tlb_impact(size, false); // Regular pages
        test_tlb_impact(size, true);  // Huge pages
    }
    
    return 0;
}