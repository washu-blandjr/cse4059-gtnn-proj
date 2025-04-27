#pragma once

// Define tile size for matrix multiplication (same as O2)
#define TILE_SIZE 32
#define WINDOW_SIZE 900

__global__ void simulate_step(
    const float* Q,
    float* vp,
    float* vn,
    int* spikesP,         // Current step spikes
    int* spikesN,         // Current step spikes
    int* spikeHistory,    // Circular buffer of total spikes
    int* currentIdx,      // Current index in circular buffer
    float* runningSum,    // Running sum of spikes in window
    int* windowFull,      // Whether the window has been filled once
    int n,
    float dt,
    float tau,
    float vmax,
    float vth,
    float Lambda,
    float C,
    bool updateHistory    // Whether to update history this step
);