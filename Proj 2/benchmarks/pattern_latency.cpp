#include "../include/memory_utils.h"
#include <windows.h> 
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>

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

double test_bandwidth_single_run(int stride, float read_ratio) {
    const size_t size = 256 * 1024 * 1024; // 256MB
    char* memory = (char*)aligned_alloc(CACHE_LINE_SIZE, size);
    
    // Initialize memory with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (size_t i = 0; i < size; i++) {
        memory[i] = static_cast<char>(dis(gen));
    }
    
    const int iterations = 100000;
    volatile char sink;
    
    // Warm-up run
    int length = 0;
    if (1000 < static_cast<int>(size/stride)) {
        length = 1000;
    } else {
        length = static_cast<int>(size/stride);
    }

    for (int i = 0; i < length; i++) {
        size_t idx = (i * stride) % size;
        if (rand() % 100 < read_ratio * 100) {
            sink = memory[idx];
        } else {
            memory[idx] = i & 0xFF;
        }
    }
    
    flush_cache(); // Add cache flush before measurement
    
    // Actual measurement
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        size_t idx = (i * stride) % size;
        if (rand() % 100 < read_ratio * 100) {
            sink = memory[idx];
        } else {
            memory[idx] = i & 0xFF;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_time_ns = std::chrono::duration<double, std::nano>(end - start).count();
    double avg_latency_ns = total_time_ns / iterations;
    
    aligned_free(memory);
    return avg_latency_ns;
}

void test_bandwidth(int stride, float read_ratio) {
    const int NUM_RUNS = 3;
    double total_latency = 0.0;
    
    for (int run = 0; run < NUM_RUNS; ++run) {
        total_latency += test_bandwidth_single_run(stride, read_ratio);
    }
    
    double avg_latency = total_latency / NUM_RUNS;
    
    std::cout << "Stride: " << stride << ", ReadRatio: " << std::fixed << std::setprecision(1) << read_ratio 
              << ", Latency: " << std::fixed << std::setprecision(2) << avg_latency << " ns" << std::endl;
}

int main() {
    set_high_priority_affinity();
    
    std::vector<int> strides = {64, 128, 256, 512, 1024, 2048, 4096};
    std::vector<float> read_ratios = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    
    // Create and randomize test order
    std::vector<std::pair<int, float>> tests;
    for (int stride : strides) {
        for (float ratio : read_ratios) {
            tests.emplace_back(stride, ratio);
        }
    }
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(tests.begin(), tests.end(), g);
    
    std::cout << "Starting pattern latency measurements..." << std::endl;
    for (const auto& test : tests) {
        test_bandwidth(test.first, test.second);
    }
    
    std::cout << "Pattern latency measurements completed." << std::endl;
    
    return 0;
}