#include "../include/memory_utils.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <windows.h>

// Set process affinity to core 0 and highest priority
void set_high_priority_affinity() {
    HANDLE process = GetCurrentProcess();
    HANDLE thread = GetCurrentThread();
    
    // Set to high priority class
    SetPriorityClass(process, HIGH_PRIORITY_CLASS);
    SetThreadPriority(thread, THREAD_PRIORITY_HIGHEST);
    
    // Set affinity to core 0
    DWORD_PTR affinityMask = 1; // Core 0
    DWORD_PTR previousAffinity = SetThreadAffinityMask(thread, affinityMask);
    
    // Set ideal processor to core 0
    SetThreadIdealProcessor(thread, 0);
    
    // Disable power throttling
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
    
    // Initialize memory
    for (size_t i = 0; i < size; ++i) {
        memory[i] = (char)(i % 256);
    }
    
    // Warm-up run
    for (size_t i = 0; i < size / 10; i += stride) {
        if (rand() % 100 < read_ratio * 100) {
            volatile char read = memory[i];
        } else {
            memory[i] = i;
        }
    }
    
    flush_cache(); // Clear cache after warm-up
    
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
    
    aligned_free(memory);
    return bandwidth;
}

void test_bandwidth(int stride, float read_ratio) {
    const int NUM_RUNS = 3;
    std::vector<double> bandwidths;
    
    for (int run = 0; run < NUM_RUNS; ++run) {
        double bandwidth = test_bandwidth_single_run(stride, read_ratio);
        bandwidths.push_back(bandwidth);
    }
    
    // Output all three measurements for plotting with error bars
    std::cout << "Stride: " << stride << "B, Read Ratio: " << read_ratio 
              << ", Bandwidths: ";
    for (size_t i = 0; i < bandwidths.size(); ++i) {
        std::cout << bandwidths[i];
        if (i < bandwidths.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << " MB/s\n";
}

int main() {
    set_high_priority_affinity();
    
    int strides[] = {64, 256, 1024};
    float ratios[] = {1.0f, 0.7f, 0.5f, 0.2f, 0.0f};
    
    // Create all test combinations
    std::vector<std::pair<int, float>> tests;
    for (int stride : strides) {
        for (float ratio : ratios) {
            tests.emplace_back(stride, ratio);
        }
    }
    
    // Randomize test order
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(tests.begin(), tests.end(), g);
    
    for (const auto& test : tests) {
        test_bandwidth(test.first, test.second);
    }
    
    return 0;
}

