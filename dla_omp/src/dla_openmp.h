#ifndef DLA_OPENMP_H
#define DLA_OPENMP_H

#include <omp.h>
#include "config.h"
#include "common_structs.h"
#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
extern void initialize_array_omp(int (*array), int height, int width, int value);

extern void randomize_particles_omp(Particle particles[], int count, int max_y, int max_x, int max_speed);

extern void simulate_omp(int (*grid));

#endif // DLA_OPENMP_H
