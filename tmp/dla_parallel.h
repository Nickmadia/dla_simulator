#ifndef DLA_PARALLEL  
#define DLA_PARALLEL 
#include <stdio.h>
#include "config.h"
#include "common_structs.h"
#include "dla_serial.h"
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

#include <cuda_runtime.h>

extern __global__ void initRNGStates(curandState *states, unsigned long long seed);

extern __global__ void init_array_d(int (*grid_d)[GRID_WIDTH],int value);

extern __device__ float random_float_d(curandState *states, int lower, int upper);

extern __device__ float random_speed_d(curandState *states, int max_speed);

extern __device__ void randomize_single_d(Particle *particle, curandState *states, int count, int max_y, int max_x, int max_speed);

extern __global__ void init_seeds(int (*grid_d)[GRID_WIDTH], curandState *state);

extern __global__ void init_particles_d(Particle *particles, curandState *states);

extern __device__ void move_particle_d(Particle *particle, curandState *states);

extern __global__ void tick(int (*grid_d)[GRID_WIDTH], Particle *particles, curandState *states, int k);

extern void simulate_parallel(int (*grid_d)[GRID_WIDTH]);

#endif // DLA_PARALLEL
