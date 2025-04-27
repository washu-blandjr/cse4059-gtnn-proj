#pragma once

__global__ void simulate_step(
    const float* Q,
    float* vp,
    float* vn,
    int* spikesP,  // Count of positive spikes
    int* spikesN,  // Count of negative spikes
    int n,
    float dt,
    float tau,
    float vmax,
    float vth,
    float Lambda,
    float C
);