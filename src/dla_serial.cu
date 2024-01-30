
#ifndef DLA_SERIAL
#define DLA_SERIAL
#include "dla_serial.h"
#include "dla_parallel.h"
void initialize_array(int (*array)[GRID_WIDTH], int height, int width, int value) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            array[i][j] = value;
        }
    }
}

__host__ float random_float(int lower, int upper)
{ 
        return (rand() % (upper - lower + 1)) + lower; 
} 
__host__ float random_speed(int max_speed){
    //returns a random direction 
    float direction =   rand()%3 -1;
    float speed =  ((float) rand()/RAND_MAX) * max_speed;
    
    return direction * speed;

}

__host__ void randomize_single(Particle *particle, int count, int max_y, int max_x,int max_speed){
    particle->x = random_float( 0, max_x);
    particle->y = random_float( 0, max_y);
    //particle->horizontal_speed = random_speed(MAX_SPEED);
    //particle->vertical_speed = random_speed(MAX_SPEED);
    particle->solid = false;
}

__host__ void randomize_particles(Particle particles[], int count, int max_y, int max_x,int max_speed) {
    //places particles randomly on the grid
    for (int i = 0; i < count; i++) {
        randomize_single(&particles[i], count, max_y, max_x, max_speed);
    }
}

__host__ void move_particle(Particle *particle ) {

    // move particle
    particle->x += random_speed(MAX_SPEED);
    particle->y += random_speed(MAX_SPEED);

    // check bounds
    if(particle->x - PARTICLE_RADIUS <= 0.0f) {
        particle->x = 0.01f + PARTICLE_RADIUS;
    }
    else if(particle->x + PARTICLE_RADIUS >= GRID_WIDTH) {
        particle->x = GRID_WIDTH - 0.01f - PARTICLE_RADIUS;
    }
    if(particle->y - PARTICLE_RADIUS <= 0.0f) {
        particle->y = 0.01f + PARTICLE_RADIUS ;
    }
    else if(particle->y + PARTICLE_RADIUS >= GRID_HEIGHT) {
        particle->y = GRID_HEIGHT - 0.01f - PARTICLE_RADIUS;
    }
}
void place_seeds(int (*grid)[GRID_WIDTH]){
    //initializes seeds on the grid
    if (SEED_POSITION == "center") {
        grid[GRID_WIDTH/2][GRID_HEIGHT/2] = 1;
    } else if ( SEED_POSITION == "random") {
       for (int i =0; i< SEED_COUNT; i++) {
          int x = (int)random_float(0,GRID_HEIGHT);
          int y =(int) random_float(0,GRID_WIDTH);
          grid[y][x] = 1;
       } 
    }
}
void simulate( int (*grid)[GRID_WIDTH]) {
    initialize_array(grid, GRID_HEIGHT, GRID_WIDTH, -1);
    place_seeds(grid);

    Particle particles[PARTICLE_COUNT];
    if(DEBUG)
    printf("initializing particles...\n");
    
    randomize_particles(particles, PARTICLE_COUNT, GRID_HEIGHT, GRID_WIDTH, MAX_SPEED);
    if(DEBUG)
    printf("starting serial simulation...\n\n");
    //simulating the passing of time with ticks(i) and moving each particle sequentially
    for ( int i = 0; i < ITERATIONS; i++) {
        for ( int k = 0; k < PARTICLE_COUNT; k ++) {
            if(particles[k].solid){
                continue;
            }
            move_particle(&particles[k]);
            
            check_hit(&particles[k], (int*)grid, i);
        }
    }
}
#endif // DLA_SERIAL