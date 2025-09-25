#include "../include/memory_utils.h"
#include <thread>
#include <vector>
#include <chrono>
#include <iostream>
#include <atomic>
#include <windows.h>
#include <algorithm>
#include <random>
#include <functional>  // Add this for std::ref

constexpr int MAX_THREADS = 16;
constexpr size_t BUFFER_SIZE = 64 * 1024 * 1024; // 64MB

// Set process affinity to core 0 and highest priority
void set_high_priority_affinity() {
    HANDLE process = GetCurrentProcess();
    HANDLE thread = GetCurrentThread();
    
    SetPriorityClass(process, HIGH_PRIORITY_CLASS);
    SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST);
    
    DWORD_PTR affinityMask = 1;
    SetThreadAffinityMask(thread, affinityMask);
    SetThreadIdealProcessor(thread, 0);
    
    PROCESS_POWER_THROTTLING_STATE powerThrottling = {};
    powerThrottling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
    powerThrottling.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    powerThrottling.StateMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
    
    SetProcessInformation(process, ProcessPowerThrottling, 
                         &powerThrottling, sizeof(powerThrottling));
}

// Worker function for concurrent memory access - remove thread_id parameter
void memory_worker(char* buffer, size_t size, int stride, std::atomic<bool>& start, 
                   std::atomic<bool>& stop, std::atomic<long long>& operations) {
    // Set worker thread affinity to core 0 as well
    HANDLE thread = GetCurrentThread();
    DWORD_PTR affinityMask = 1;
    SetThreadAffinityMask(thread, affinityMask);
    
    while (!start) { 
        std::this_thread::yield(); 
    }
    
    long long local_ops = 0;
    size_t i = 0;
    while (!stop) {
        buffer[i % size] = (char)(i % 256); // Write operation
        local_ops++;
        i += stride;
        if (i >= size) i = 0;
    }
    
    operations += local_ops;
}

std::pair<std::vector<double>, std::vector<double>> loaded_latency_test_single_run(int num_threads, int stride) {
    char* buffer = (char*)aligned_alloc(CACHE_LINE_SIZE, BUFFER_SIZE);
    
    // Initialize buffer
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
        buffer[i] = (char)(i % 256);
    }
    
    std::atomic<bool> start(false);
    std::atomic<bool> stop(false);
    std::atomic<long long> total_operations(0);
    std::vector<std::thread> threads;
    
    // Create worker threads - use lambda to avoid complex parameter passing
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            memory_worker(buffer, BUFFER_SIZE, stride, start, stop, total_operations);
        });
    }
    
    // Warm-up: let threads run briefly
    start = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    stop = true;
    
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    // Reset for actual measurement
    total_operations = 0;
    stop = false;
    threads.clear();
    
    // Recreate worker threads
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&]() {
            memory_worker(buffer, BUFFER_SIZE, stride, start, stop, total_operations);
        });
    }
    
    // Actual measurement
    auto test_duration = std::chrono::milliseconds(500);
    
    start = true;
    auto start_time = std::chrono::high_resolution_clock::now();
    std::this_thread::sleep_for(test_duration);
    stop = true;
    auto end_time = std::chrono::high_resolution_clock::now();
    
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    double elapsed_ns = std::chrono::duration<double, std::nano>(end_time - start_time).count();
    double throughput = (total_operations * stride) / (elapsed_ns / 1e9); // Bytes per second
    double avg_latency = (total_operations > 0) ? (elapsed_ns / total_operations) : 0.0; // Nanoseconds per operation
    
    aligned_free(buffer);
    return std::make_pair(std::vector<double>{throughput}, std::vector<double>{avg_latency});
}

void loaded_latency_test(int num_threads, int stride) {
    const int NUM_RUNS = 3;
    std::vector<double> throughputs;
    std::vector<double> latencies;
    
    for (int run = 0; run < NUM_RUNS; ++run) {
        // Warm-up run
        loaded_latency_test_single_run(num_threads, stride);
        flush_cache();
        
        auto result = loaded_latency_test_single_run(num_threads, stride);
        throughputs.insert(throughputs.end(), result.first.begin(), result.first.end());
        latencies.insert(latencies.end(), result.second.begin(), result.second.end());
        
        // Small delay between runs
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Output all three measurements
    std::cout << "Threads: " << num_threads << ", Stride: " << stride 
              << ", Throughputs: ";
    for (size_t i = 0; i < throughputs.size(); ++i) {
        std::cout << (throughputs[i] / (1024*1024)) << " MB/s";
        if (i < throughputs.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ", Latencies: ";
    for (size_t i = 0; i < latencies.size(); ++i) {
        std::cout << latencies[i] << " ns/op";
        if (i < latencies.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << std::endl;
}

int main() {
    set_high_priority_affinity();
    
    // Create randomized test order
    std::vector<std::pair<int, int>> tests;
    for (int threads = 1; threads <= MAX_THREADS; threads *= 2) {
        int strides[] = {64, 256, 1024};
        for (int stride : strides) {
            tests.emplace_back(threads, stride);
        }
    }
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(tests.begin(), tests.end(), g);
    
    std::cout << "=== Loaded Latency Tests ===" << std::endl;
    for (const auto& test : tests) {
        loaded_latency_test(test.first, test.second);
    }
    
    return 0;
}
