#ifndef PARTICLE_SIMULATION_H
#define PARTICLE_SIMULATION_H

#include "common_structs.h"
#include "config.h"
void initialize_array(int (*array)[GRID_WIDTH], int height, int width, int value);

__host__ float random_float(int lower, int upper);

__host__ float random_speed(int max_speed);

__host__ void randomize_single(Particle *particle, int count, int max_y, int max_x, int max_speed);

__host__ void randomize_particles(Particle particles[], int count, int max_y, int max_x, int max_speed);

__host__ void move_particle(Particle *particle);

void place_seeds(int (*grid)[GRID_WIDTH]);

void simulate( int (*grid)[GRID_WIDTH]);

#endif //PARTICLE_SIMULATION_H
