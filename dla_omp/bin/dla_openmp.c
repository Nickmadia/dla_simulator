#include <string.h>
#include "dla_openmp.h"
#include <math.h>
#include <time.h>
#include <unistd.h>
pthread_barrier_t barrier; 
void initialize_array_omp(int *array, int height, int width, int value) {
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
void* thread_ticks(void* arg) {
    ThreadInfo *info = (ThreadInfo*)arg;

    srand(info->thread_id * 123421);

    for (int i = 0; i < ITERATIONS; i++) {
        for (int k = info->start_index; k < info->end_index; k++) {
            if (!info->particles[k].solid) {
                move_particle_omp(&info->particles[k]);
                int x = info->particles[k].x;
                int y = info->particles[k].y;

                if (check_hit_omp(x, y, info->grid)) {
                    aggregate(&info->particles[k], info->grid, i);
                }
            }

        }
        pthread_barrier_wait(&barrier);
    }

    pthread_exit(NULL);
}
void ticks(int (*grid), Particle * particles, int nt) {
    pthread_t threads[nt];
    ThreadInfo thread_info[nt];
    int chunk = PARTICLE_COUNT / nt;
    pthread_barrier_init(&barrier, NULL, nt);    
    printf("chunk size - %d\n",chunk);

    for (int i = 0; i < nt; i++) {
        thread_info[i].thread_id = i;
        thread_info[i].start_index = i * chunk;
        thread_info[i].end_index = (i == nt - 1) ? PARTICLE_COUNT : (i + 1) * chunk;
        thread_info[i].grid = grid;
        thread_info[i].particles = particles;

        pthread_create(&threads[i], NULL, thread_ticks, (void*)&thread_info[i]);
    }

    for (int i = 0; i < nt; i++) {
        pthread_join(threads[i], NULL);
    }
    pthread_barrier_destroy(&barrier); 
    
}

void simulate_omp(int *grid) {
    initialize_array_omp(grid, GRID_HEIGHT, GRID_WIDTH, -1);
        
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
    struct timeval start, end;

    gettimeofday(&start, NULL);
    //int grid[GRID_HEIGHT][GRID_WIDTH ];
    int * grid = malloc(sizeof(int) * GRID_HEIGHT * GRID_WIDTH);

    simulate_omp((int*)grid);

    gettimeofday(&end, NULL);

    double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
    printf("Elapsed time = %f\n", elapsed_time);
    save_heat_map_to_binary_file(grid, "../omp_simulation.bin");
    free(grid);
}