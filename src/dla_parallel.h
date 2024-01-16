// dla_parallel.h

#ifndef DLA_PARALLEL_H
#define DLA_PARALLEL_H

#include <curand_kernel.h>
#include "config.h"
#include "common_structs.h"
#include <stdio.h>

__device__ float random_float_d(curandState *states, int lower, int upper);

__device__ float random_speed_d(curandState *states, int max_speed);

__global__ void initRNGStates(curandState *states, unsigned long long seed);

__global__ void init_array_d(int value);

__device__ void randomize_single_d(Particle *particle, curandState *states, int count, int max_y, int max_x, int max_speed);

__global__ void init_seeds(int * grid_d, curandState *state);

__global__ void init_particles_d(Particle *particles, curandState *states);

__device__ void move_particle_d(Particle *particle, curandState *states);

__device__ __host__ void make_static(Particle *particle, int *grid, int tick);

__device__ __host__ void check_hit(Particle *particle, int *grid, int tick);

__global__ void tick(int * grid_d, Particle *particles, curandState *states, int k);

void simulate_parallel(int * grid);

#endif // DLA_PARALLEL_H
