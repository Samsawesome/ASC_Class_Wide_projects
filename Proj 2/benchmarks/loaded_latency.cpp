#include "../include/memory_utils.h"
#include <thread>
#include <vector>
#include <chrono>
#include <iostream>
#include <atomic>

constexpr int MAX_THREADS = 16;
constexpr size_t BUFFER_SIZE = 64 * 1024 * 1024; // 64MB

// Worker function for concurrent memory access
void memory_worker(char* buffer, size_t size, int stride, std::atomic<bool>& start, std::atomic<long long>& operations) {
    while (!start) { std::this_thread::yield(); } // Wait for start signal
    
    long long local_ops = 0;
    for (size_t i = 0; i < size; i += stride) {
        buffer[i] = i; // Write operation
        local_ops++;
    }
    
    operations += local_ops;
}

void loaded_latency_test(int num_threads, int stride) {
    char* buffer = (char*)aligned_alloc(CACHE_LINE_SIZE, BUFFER_SIZE);
    
    std::atomic<bool> start(false);
    std::atomic<long long> total_operations(0);
    std::vector<std::thread> threads;
    
    // Create worker threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(memory_worker, buffer, BUFFER_SIZE, stride, 
                            std::ref(start), std::ref(total_operations));
    }
    
    // Let threads run for a fixed duration
    auto test_duration = std::chrono::milliseconds(1000);
    
    start = true; // Start all threads
    auto start_time = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(test_duration);
    start = false; // Signal threads to stop
    
    // Wait for all threads to complete
    for (auto& t : threads) {
        t.join();
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double elapsed_ns = std::chrono::duration<double, std::nano>(end_time - start_time).count();
    double throughput = (total_operations * stride) / (elapsed_ns / 1e9); // Bytes per second
    double avg_latency = elapsed_ns / total_operations; // Nanoseconds per operation
    
    std::cout << "Threads: " << num_threads << ", Stride: " << stride 
              << ", Throughput: " << throughput / (1024*1024) << " MB/s"
              << ", Latency: " << avg_latency << " ns/op" << std::endl;
    
    aligned_free(buffer);
}

int main() {
    std::cout << "=== Loaded Latency Tests ===" << std::endl;
    
    // Test different levels of concurrency
    for (int threads = 1; threads <= MAX_THREADS; threads *= 2) {
        // Test different access granularities
        int strides[] = {64, 256, 1024};
        for (int stride : strides) {
            loaded_latency_test(threads, stride);
        }
    }
    
    return 0;
}