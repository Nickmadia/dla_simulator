#include <string.h>
#include "dla_openmp.h"
#include <math.h>
#include <time.h>
#include <unistd.h>
#define GRID_PADDING ((64/ sizeof(int)) - GRID_WIDTH % (64/ sizeof(int)))
void initialize_array_omp(int *array, int height, int width, int value) {
    #pragma omp parallel for schedule(static, 16) 
        for (int i = 0; i< width*height; i++) {
            array[i] = value;
        }
}
//TODO pass just x and y
void make_static_omp(Particle *particle, int *grid, int tick) {
    //TODO implement circle check instead of squared
    for ( int i = -PARTICLE_RADIUS; i <= PARTICLE_RADIUS; i++) {
        for (int j = -PARTICLE_RADIUS; j<= PARTICLE_RADIUS; j++) {
            int dx = (int)particle->x - j;
            int dy = (int)particle->y - i;
            // if in bound
            //float da = (float) particle-> x - dx;
            //float db = (float) particle-> y - dy;
            if (dx >= 0 && dx < GRID_WIDTH && dy >= 0 && dy < GRID_HEIGHT) {
                float d = sqrt((float)(i * i) +(float) (j *j));
                if(d<= PARTICLE_RADIUS) {
                   #pragma omp critical
                   grid[dy* GRID_WIDTH + dx] = tick; 

                }
            }

        }
    }
    particle->solid = true;
}
void aggregate(Particle * particle, int * grid, int tick){
    grid[particle->y * GRID_WIDTH + particle->x] = tick;
    particle->solid = true;
}
//dont pass the particle to limit memory reads since we just need x and y
bool check_hit_omp(int x , int y, int * grid ) {
    for ( int i = -PARTICLE_RADIUS; i <= PARTICLE_RADIUS ; i++) {
        for (int j = -PARTICLE_RADIUS; j<= PARTICLE_RADIUS; j++) {
            int dx = x - j;
            int dy = y - i;
            // if in bound
            if (dx >= 0 && dx < GRID_WIDTH && dy >= 0 && dy < GRID_HEIGHT) {
                //float da = (float) particle-> x - dx;
                //float db = (float) particle-> y - dy;
                //float d = sqrt((float)(i*i) + (float)(j *j));
                //if(d<= PARTICLE_RADIUS) {
                
                if (grid[dy* GRID_WIDTH + dx] >= 0) {
                    return true;
                }
                //}
            }
        }
    }
    return false;
}
int random_float_omp(int lower, int upper)
{ 
        return (rand() % (upper - lower + 1)) + lower; 
} 
int random_speed_omp(int max_speed){
    int direction =  rand() % 3 -1;
    int speed =   rand() % (max_speed + 1);
    
    return direction * speed;

}

void randomize_single_omp(Particle *particle, int count, int max_y, int max_x,int max_speed){
    particle->x = rand() % GRID_WIDTH;
    particle->y = rand() % GRID_HEIGHT;
    //particle->horizontal_speed = random_speed(MAX_SPEED);
    //particle->vertical_speed = random_speed(MAX_SPEED);
    particle->solid = false;
}

void randomize_particles_omp(Particle particles[], int count, int max_y, int max_x, int max_speed) {
    #pragma omp parallel for schedule(static,8) shared(particles)
    for (int i = 0; i < count; i++) {
        randomize_single_omp(&particles[i], count, max_y, max_x, max_speed);
    }

}

void move_particle_omp(Particle *particle ) {

    // move particle
    particle->x += random_speed_omp(MAX_SPEED);
    particle->y += random_speed_omp(MAX_SPEED);

    // check bounds using min max
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
void place_seeds(int (*grid)){

    if (SEED_POSITION == "center") {
        grid[(GRID_WIDTH * (GRID_HEIGHT/2) ) + (GRID_WIDTH/2)] = 1;
    } else if ( SEED_POSITION == "random") {
       for (int i =0; i< SEED_COUNT; i++) {
          int y = (int)random_float_omp(0,GRID_HEIGHT);
          int x =(int) random_float_omp(0,GRID_WIDTH);
          grid[(y * GRID_WIDTH) + x] = 1;
       } 
    }
}
void ticks(int (*grid), Particle * particles, int nt) {

    omp_set_num_threads(nt);
    int num_threads = nt;
    int chunk = PARTICLE_COUNT/num_threads;
    int * t_grid = malloc(GRID_HEIGHT * GRID_WIDTH * sizeof(int));
    printf("chunk size - %d\n",chunk);
    //memcpy(t_grid, (int*)grid, GRID_HEIGHT * GRID_WIDTH * sizeof(int));
    #pragma omp parallel
    {
        srand(omp_get_thread_num()*123421);
    #pragma omp for collapse(2)  schedule (static, chunk) ordered
    for (int i =0; i< ITERATIONS; i++) {
        for (int k = 0; k < PARTICLE_COUNT; k++) {
            //int ti0d = omp_get_thread_num();

            #pragma omp ordered
            if (!particles[k].solid) {
                move_particle_omp(&particles[k]);
                int x = particles[k].x;
                int y = particles[k].y;
                //usleep(1);
                //if(particles[k].y >= chunk * tid-1 && particles[k].y < chunk *(tid ))
                //try changing check hit to assign different parts of the grid to each thread
                if (check_hit_omp(x,y, (int*)grid)){
                    make_static_omp(&particles[k], grid,i);
                }

            }
            }

    }
    }
}
void spam(){
    int tid = omp_get_thread_num();
    long test = 100000 ;
    int chunk = test/ omp_get_thread_num();
    #pragma omp parallel for
    for(int i = 0; i<test;i ++){
        
        usleep(10) ;
         
    }
    
}
void simulate_omp(int *grid) {
    omp_set_num_threads(NUM_T);
    double start_parallel = omp_get_wtime();
    initialize_array_omp(grid, GRID_HEIGHT, GRID_WIDTH, -1);
    double end_parallel = omp_get_wtime();
        
    double elapsed_time_parallel  = (double)(end_parallel - start_parallel)  ;
    printf("%f\n", elapsed_time_parallel);
    printf("init threads\n");
    place_seeds(grid);
    Particle particles[PARTICLE_COUNT];


    printf("init particles\n");
    randomize_particles_omp(particles, PARTICLE_COUNT, GRID_HEIGHT, GRID_WIDTH, MAX_SPEED);
    ticks(grid, particles,NUM_T );
    //#pragma omp parallel
    //spam();

}
void save_heat_map_to_binary_file(int *heatmap, const char *filename) {
    FILE *file = fopen(filename, "wb");
    if (file != NULL) {
        fwrite(heatmap, sizeof(int), GRID_HEIGHT* GRID_WIDTH, file);
        fclose(file);
    } else {
        printf("Error opening file for writing.\n");
    }
}
int main() {
    //int grid[GRID_HEIGHT][GRID_WIDTH ];
    int * grid = malloc(sizeof(int) * GRID_HEIGHT * GRID_WIDTH);
    double start_parallel = omp_get_wtime();

    simulate_omp((int*)grid);

    double end_parallel = omp_get_wtime();
    
    double elapsed_time_parallel  = (double)(end_parallel - start_parallel)  ;
    printf("Elapsed time = %f\n", elapsed_time_parallel);
    save_heat_map_to_binary_file(grid, "../omp_simulation.bin");
    free(grid);
}