#include <stdio.h>
#include <stdlib.h>
#include "config.h"
#include "common_structs.h"
#include "dla_serial.h"
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include "dla_parallel.h"

void save_heat_map_to_binary_file(int heatmap[GRID_HEIGHT][GRID_WIDTH], const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (file != NULL) {
        fwrite(heatmap, sizeof(int), GRID_HEIGHT* GRID_WIDTH, file);
        fclose(file);
    } else {
        printf("Error opening file for writing.\n");
    }
}

void print_parameters() {

    printf("Current parameters: \n\n");
    printf("grid height: %d\n", GRID_HEIGHT);
    printf("grid width: %d\n", GRID_WIDTH);
    printf("iteration count: %d\n", ITERATIONS);
    printf("seed position: %s\n", SEED_POSITION);
    printf("seed count: %d\n", SEED_COUNT);
    printf("particle count: %d\n", PARTICLE_COUNT);
    printf("particle radius: %d\n", PARTICLE_RADIUS);
    printf("particle max_speed: %d\n\n", MAX_SPEED);
}
// Function to initialize RNG states
int main() {

    print_parameters();

    srand((unsigned int)time(NULL));
    //declaring array of ints in order to use a heat map later
    int * grid = (int*)malloc(GRID_HEIGHT * GRID_WIDTH * sizeof(int));
    if (!SKIP_SERIAL) {

        time_t start_time = clock();
        if (DEBUG) {
            printf("initializing grid...\n");
    
        }
        simulate((int (*)[GRID_WIDTH])grid); 

        time_t end_time = clock();
        double elapsed_time = (double)(end_time - start_time)/CLOCKS_PER_SEC;

        printf("serial Elapsed Time: %f seconds\n\n", elapsed_time);

        save_heat_map_to_binary_file((int (*)[GRID_WIDTH])grid, "serial_simulation.bin");

    }
    //start parallel 
    time_t start_parallel = clock();

    simulate_parallel((int*)grid); 

    time_t end_parallel = clock();
    
    double elapsed_time_parallel  = (double)(end_parallel - start_parallel) / CLOCKS_PER_SEC;

    printf("parallel elapsed Time: %f seconds\n\n", elapsed_time_parallel);

    save_heat_map_to_binary_file((int (*)[GRID_WIDTH])grid, "parallel_simulation.bin");
    free(grid);
    return 0;
}