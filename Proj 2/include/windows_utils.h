#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <iostream>
#include <vector>

class WindowsPerformanceCounter {
public:
    WindowsPerformanceCounter(const char* counterPath) {
        PdhOpenQuery(NULL, NULL, &hQuery);
        PdhAddCounterA(hQuery, counterPath, 0, &hCounter);
        PdhCollectQueryData(hQuery);
    }
    
    double getValue() {
        PDH_FMT_COUNTERVALUE value;
        PdhCollectQueryData(hQuery);
        PdhGetFormattedCounterValue(hCounter, PDH_FMT_DOUBLE, NULL, &value);
        return value.doubleValue;
    }
    
    ~WindowsPerformanceCounter() {
        PdhCloseQuery(hQuery);
    }

private:
    PDH_HQUERY hQuery;
    PDH_HCOUNTER hCounter;
};