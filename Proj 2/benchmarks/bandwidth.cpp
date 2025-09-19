#include "../include/memory_utils.h"
#include <chrono>
#include <iostream>

void test_bandwidth(int stride, float read_ratio) {
    const size_t size = 256 * 1024 * 1024; // 256MB
    char* memory = (char*)aligned_alloc(CACHE_LINE_SIZE, size);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < size; i += stride) {
        if (rand() % 100 < read_ratio * 100) {
            volatile char read = memory[i];
        } else {
            memory[i] = i;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double bandwidth = (size / (1024.0 * 1024.0)) / (time_ms / 1000.0); // MB/s
    
    std::cout << "Stride: " << stride << "B, Read Ratio: " << read_ratio 
              << ", Bandwidth: " << bandwidth << " MB/s\n";
    
    aligned_free(memory);
}

int main() {
    int strides[] = {64, 256, 1024};
    float ratios[] = {1.0f, 0.7f, 0.5f, 0.0f};
    
    for (int stride : strides) {
        for (float ratio : ratios) {
            test_bandwidth(stride, ratio);
        }
    }
}