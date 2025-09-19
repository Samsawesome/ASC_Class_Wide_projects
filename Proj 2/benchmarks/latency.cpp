#include "../include/memory_utils.h"
#include <chrono>
#include <iostream>

constexpr int ITERATIONS = 1000000;

template<typename T>
double measure_latency(T* memory, int size) {
    flush_cache();
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        memory = (T*)*memory;
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double, std::nano>(end - start).count() / ITERATIONS;
}

int main() {
    // Test different memory hierarchy levels
    const int sizes[] = { // Approximate sizes for different cache levels
        32 * 1024,    // L1
        256 * 1024,   // L2
        8 * 1024 * 1024, // L3
        64 * 1024 * 1024 // DRAM
    };

    for (int size : sizes) {
        int* memory = (int*)aligned_alloc(CACHE_LINE_SIZE, size);
        // Create pointer-chasing pattern
        for (int i = 0; i < size / sizeof(int); ++i) {
            memory[i] = (i + 64/sizeof(int)) % (size/sizeof(int));
        }
        
        double latency = measure_latency(memory, size);
        std::cout << "Size: " << size << " bytes, Latency: " << latency << " ns\n";
        
        aligned_free(memory);
    }
}