#include <stdio.h>
#include "config.h"
#include "common_structs.h"
#include <math.h>
#include <time.h>
void initializeArray(int (*array)[GRID_WIDTH], int height, int width, int value) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            array[i][j] = value;
        }
    }
}
//random float in range
float random_float(int lower, int upper)
{ 
        return (rand() % (upper - lower + 1)) + lower; 
} 
float random_speed(int max_speed){
    float direction =  ((float) rand()/RAND_MAX) * M_PI * 2;
    float speed =  ((float) rand()/RAND_MAX) * max_speed;
    
    return sinf(direction) * speed;

}
void randomize_single(Particle *particle, int count, int max_y, int max_x,int max_speed){
    particle->x = random_float( 0, max_x);
    particle->y = random_float( 0, max_y);
    //particle->horizontal_speed = random_speed(MAX_SPEED);
    //particle->vertical_speed = random_speed(MAX_SPEED);
    particle->solid = false;
}
void randomize_particles(Particle particles[], int count, int max_y, int max_x,int max_speed) {
    // init trough clock time
    //TODO encapsulate rnadomization of a single particle
    for (int i = 0; i < count; i++) {
        randomize_single(&particles[i], count, max_y, max_x, max_speed);
    }
}
void move_particle(Particle *particle ) {

    particle->horizontal_speed = random_speed(MAX_SPEED);
    particle->vertical_speed = random_speed(MAX_SPEED);
    // move particle
    particle->x += particle->horizontal_speed;
    particle->y += particle->vertical_speed;

    // check bounds
    if(particle->x - PARTICLE_RADIUS <= 0.0f) {
        particle->x = 0.01f + PARTICLE_RADIUS;
        particle->horizontal_speed *= -1.0f;
    }
    else if(particle->x + PARTICLE_RADIUS >= GRID_WIDTH) {
        particle->x = GRID_WIDTH - 0.01f - PARTICLE_RADIUS;
        particle->horizontal_speed *= -1.0f;
    }
    if(particle->y - PARTICLE_RADIUS <= 0.0f) {
        particle->y = 0.01f + PARTICLE_RADIUS ;
        particle->vertical_speed *= -1.0f;
    }
    else if(particle->y + PARTICLE_RADIUS >= GRID_HEIGHT) {
        particle->y = GRID_HEIGHT - 0.01f - PARTICLE_RADIUS;
        particle->vertical_speed *= -1.0f;
    }
}
void make_static(Particle *particle, int (*grid)[GRID_WIDTH], int tick) {
    for ( int i = -PARTICLE_RADIUS; i <= PARTICLE_RADIUS; i++) {
        for (int j = -PARTICLE_RADIUS; j<= PARTICLE_RADIUS; j++) {
            int dx = (int)particle->x - i;
            int dy = (int)particle->y - j;
            // if in bound
            if (dx >= 0 && dx < GRID_WIDTH && dy >= 0 && dy < GRID_HEIGHT) {
                   grid[dy][dx] = tick; 
            }
        }
    }
    particle->solid = true;
}
void check_hit(Particle *particle, int (*grid)[GRID_WIDTH], int tick) {
    bool loop = true;
    for ( int i = -PARTICLE_RADIUS; i <= PARTICLE_RADIUS && loop; i++) {
        for (int j = -PARTICLE_RADIUS; j<= PARTICLE_RADIUS; j++) {
            int dx = particle->x - i;
            int dy = particle->y - j;
            // if in bound
            if (dx >= 0 && dx < GRID_WIDTH && dy >= 0 && dy < GRID_HEIGHT) {
                
                if (grid[(int)dy][(int)dx] >= 0) {
                    make_static(particle, grid, tick);
                    loop = false;
                    break;
                }
            }
        }
    }
}
void simulate(Particle particles[], int (*grid)[GRID_WIDTH]) {
    for ( int i = 0; i < ITERATIONS; i++) {
        for ( int k = 0; k < PARTICLE_COUNT; k ++) {
            if(particles[k].solid){
                continue;
            }
            move_particle(&particles[k]);
            check_hit(&particles[k], grid, i);
        }
    }
}
void save_heat_map_to_binary_file(int heatmap[GRID_HEIGHT][GRID_WIDTH], const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (file != NULL) {
        fwrite(heatmap, sizeof(int), GRID_HEIGHT* GRID_WIDTH, file);
        fclose(file);
    } else {
        printf("Error opening file for writing.\n");
    }
}
int main() {
    printf("grid height: %d\n", GRID_HEIGHT);
    printf("grid width: %d\n", GRID_WIDTH);
    printf("iteration count: %d\n", ITERATIONS);
    printf("seed position: %s\n", SEED_POSITION);
    printf("seed count: %d\n", SEED_COUNT);
    printf("particle count: %d\n\n", PARTICLE_COUNT);


    srand((unsigned int)time(NULL));
    //declaring array of ints in order to use a heat map later
    int grid[GRID_HEIGHT][GRID_WIDTH];
    time_t start_time = clock();
    if (DEBUG) {
        printf("initializing grid...\n");
    
    }
    //init array to -1 
    initializeArray(grid, GRID_HEIGHT, GRID_WIDTH, -1);
    
    //place static seed
    if (SEED_POSITION == "center") {
        grid[GRID_WIDTH/2][GRID_HEIGHT/2] = 1;
        //grid[10][10] = 1;
    } else if ( SEED_POSITION == "random") {
       for (int i =0; i< SEED_COUNT; i++) {
          int x = (int)random_float(0,GRID_HEIGHT);
          int y =(int) random_float(0,GRID_WIDTH);
          grid[y][x] = 1;
       } 
    }
    //TODO handle other positions

    //init particles array
    Particle particleArray[PARTICLE_COUNT];
    printf("initializing particles...\n");
    randomize_particles(particleArray, PARTICLE_COUNT, GRID_HEIGHT, GRID_WIDTH, MAX_SPEED);
    printf("starting serial simulation...\n\n");
    simulate(particleArray, grid); 
    time_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time)/CLOCKS_PER_SEC;

    // Print the elapsed time
    printf("Elapsed Time: %f seconds\n\n", elapsed_time);

    //write movement function
    printf("saving binary file...\n");
    save_heat_map_to_binary_file(grid, "simulation.bin");
     
    //save simulation results
    return 0;
}