#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include "cuda_runtime.h"


__global__ void computeAccels(vector3* accels, const vector3* hPos, const double* mass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < NUMENTITIES && j < NUMENTITIES) {
        if (i == j) {
            FILL_VECTOR(accels[i * NUMENTITIES + j], 0, 0, 0);
        } else {
            vector3 distance;
            for (int k = 0; k < 3; k++) 
                distance[k] = hPos[i*3 + k] - hPos[j*3 + k];

            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;

            FILL_VECTOR(accels[i * NUMENTITIES + j],
                        accelmag * distance[0] / magnitude,
                        accelmag * distance[1] / magnitude,
                        accelmag * distance[2] / magnitude);
        }
    }
    return;
}

__global__ void sumColumnsAndApplyAcceleration(vector3* accels, vector3* hVel, vector3* hPos, const double* mass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NUMENTITIES) {
        vector3 accel_sum = {0, 0, 0};

        for (int j = 0; j < NUMENTITIES; j++) {
            accel_sum[0] += accels[j * NUMENTITIES + i][0];
            accel_sum[1] += accels[j * NUMENTITIES + i][1];
            accel_sum[2] += accels[j * NUMENTITIES + i][2];
        }

        // Compute the new velocity based on the acceleration and time interval
        // Compute the new position based on the velocity and time interval
        for (int k = 0; k < 3; k++) {
            hVel[i][k] += accel_sum[k] * INTERVAL;
            hPos[i][k] += hVel[i][k] * INTERVAL;
        }
    }
    return;
}

void compute() {
    // Assuming hPos, hVel, and mass are already declared and allocated on the host.
    // Allocate memory for device counterparts and copy data from host to device.
    
    // allocate mem
    vector3* dPos;
    cudaMalloc((void**)&dPos, sizeof(double) * NUMENTITIES * 3);
    vector3* dVel;
    cudaMalloc((void**)&dVel, sizeof(double) * NUMENTITIES * 3);
    double* dMass;
    cudaMalloc((void**)&dMass, sizeof(double) * NUMENTITIES);
    vector3* dAccels;
    cudaMalloc((void**)&dAccels, sizeof(vector3) * NUMENTITIES * NUMENTITIES);

    //copy to devices
    cudaMemcpy(dVel, hVel, sizeof(double) * NUMENTITIES * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(dPos, hPos, sizeof(double) * NUMENTITIES * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(dMass, mass, sizeof(double) * NUMENTITIES, cudaMemcpyHostToDevice);

    dim3 blockSizeAccels(16, 16);
    dim3 gridSizeAccels((NUMENTITIES + blockSizeAccels.x - 1) / blockSizeAccels.x,
                        (NUMENTITIES + blockSizeAccels.y - 1) / blockSizeAccels.y);

    computeAccels<<<gridSizeAccels, blockSizeAccels>>>(dAccels, dPos, dMass);
    cudaDeviceSynchronize();

    dim3 blockSizeSum(256);
    dim3 gridSizeSum((NUMENTITIES + blockSizeSum.x - 1) / blockSizeSum.x);

    sumColumnsAndApplyAcceleration<<<gridSizeSum, blockSizeSum>>>(dAccels, dVel, dPos, dMass);
    cudaDeviceSynchronize();

    // Copy the results back to the host if needed
    cudaMemcpy(hPos, dPos, sizeof(double) * NUMENTITIES * 3 * NUMENTITIES, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, dVel, sizeof(double) * NUMENTITIES * 3 * NUMENTITIES, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dPos);
    cudaFree(dVel);
    cudaFree(dMass);
    cudaFree(dAccels);
}
