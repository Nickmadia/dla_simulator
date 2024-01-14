
#ifndef DLA_SERIAL
#define DLA_SERIAL
#include "dla_serial.h"
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
    float direction =  ((float) rand()/RAND_MAX) * M_PI * 2;
    float speed =  ((float) rand()/RAND_MAX) * max_speed;
    
    return sinf(direction) * speed;

}

__host__ void randomize_single(Particle *particle, int count, int max_y, int max_x,int max_speed){
    particle->x = random_float( 0, max_x);
    particle->y = random_float( 0, max_y);
    //particle->horizontal_speed = random_speed(MAX_SPEED);
    //particle->vertical_speed = random_speed(MAX_SPEED);
    particle->solid = false;
}

__host__ void randomize_particles(Particle particles[], int count, int max_y, int max_x,int max_speed) {
    // init trough clock time
    //TODO encapsulate rnadomization of a single particle
    for (int i = 0; i < count; i++) {
        randomize_single(&particles[i], count, max_y, max_x, max_speed);
    }
}

__host__ void move_particle(Particle *particle ) {

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
#endif // DLA_SERIAL