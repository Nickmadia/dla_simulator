#include <stdio.h>
#include "config.h"
#include "common_structs.h"
#include "dla_serial.h"
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>

__device__ int grid_d[GRID_HEIGHT][GRID_WIDTH];

__global__ void initRNGStates(curandState *states, unsigned long long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<PARTICLE_COUNT) {
        curand_init(seed, tid, 0, &states[tid]);
    }
   
}

__global__ void init_array_d( int value) {
    grid_d[blockIdx.y * blockDim.y + threadIdx.y][blockIdx.x * blockDim.x + threadIdx.x] = value;
}
//random float in range
__device__ float random_float_d(curandState *states, int lower, int upper)
{ 
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    return (curand_uniform(states + tid) * (upper - lower + 1)) + lower; 
} 
__device__ float random_speed_d(curandState *states, int max_speed){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float direction =  ((float) curand_uniform(states + tid)) * M_PI * 2;
    float speed =  ((float) curand_uniform(states + tid)) * max_speed;
    
    return sinf(direction) * speed;

}
__device__ void randomize_single_d(Particle *particle, curandState * states,int count, int max_y, int max_x,int max_speed){
    particle->x = random_float_d( states, 0, max_x);
    particle->y = random_float_d( states, 0, max_y);
    //particle->horizontal_speed = random_speed(MAX_SPEED);
    //particle->vertical_speed = random_speed(MAX_SPEED);
    particle->solid = false;
}
__global__ void init_seeds( curandState *state) {
    if (SEED_POSITION == "center") {
        grid_d[GRID_HEIGHT/2][GRID_WIDTH/2] = 1;
        //grid[10][10] = 1;
    } else if ( SEED_POSITION == "random") {
       for (int i =0; i< SEED_COUNT; i++) {
          int x = (int)random_float_d(state, 0,GRID_HEIGHT);
          int y =(int) random_float_d(state, 0,GRID_WIDTH);
          grid_d[y][x] = 1;
       } 
    }
}
__global__ void init_particles_d(Particle* particles, curandState *states) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // particle index in the particles array
    randomize_single_d(&particles[i], states, PARTICLE_COUNT, GRID_HEIGHT, GRID_WIDTH, MAX_SPEED);
}
__device__ void move_particle_d(Particle *particle, curandState* states ) {

    particle->horizontal_speed = random_speed_d(states, MAX_SPEED);
    particle->vertical_speed = random_speed_d(states, MAX_SPEED);
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
__device__ __host__ void make_static(Particle *particle, int (*grid)[GRID_WIDTH], int tick) {
    //TODO implement circle check instead of squared
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
__device__ __host__ void check_hit(Particle *particle, int (*grid)[GRID_WIDTH], int tick) {
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


__global__ void tick( Particle *particles, curandState * states, int k) {

    int i = threadIdx.x + blockIdx.x * blockDim.x; 
    Particle *particle = particles + i;
    if (i < PARTICLE_COUNT) {
        if (!particle->solid) {
            move_particle_d(particle, states);
            check_hit(particle,grid_d,k);
        }
    }
}
void simulate_parallel() {
    int threads_block = 16;
    int sim_blocks = (PARTICLE_COUNT)  / threads_block ;
    //init random generator
    curandState *d_rngStates;
    cudaMalloc((void**)&d_rngStates, PARTICLE_COUNT * sizeof(curandState));
    
    int seed = rand();
    initRNGStates<<<sim_blocks,threads_block>>>(d_rngStates, seed);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "initRNGStates launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    init_seeds<<<1,1>>>( d_rngStates);

    cudaDeviceSynchronize();
    //init particles array
    size_t mem_size = PARTICLE_COUNT * sizeof(Particle);
    Particle* particles_d;
    cudaMalloc(&particles_d, mem_size);
    // init particles parallel
    init_particles_d<<<sim_blocks,threads_block>>>(particles_d, d_rngStates);
    cudaDeviceSynchronize();
    printf("starting parallel simulation...\n\n");
    
    for( int i = 0; i< ITERATIONS; i++) {
        tick<<<sim_blocks,threads_block>>>( particles_d, d_rngStates, i);
        cudaDeviceSynchronize();
    }
    cudaFree(d_rngStates);
    cudaFree(particles_d);
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


// Function to initialize RNG states
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
    //initialize_array(grid, GRID_HEIGHT, GRID_WIDTH, -1);
    
    //place static seed
    if (SEED_POSITION == "center") {
        //grid[GRID_WIDTH/2][GRID_HEIGHT/2] = 1;
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
    //simulate(particleArray, grid); 
    time_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time)/CLOCKS_PER_SEC;

    // Print the elapsed time
    printf("serial Elapsed Time: %f seconds\n\n", elapsed_time);

    printf("saving binary file...\n");
    save_heat_map_to_binary_file(grid, "serial_simulation.bin");

    //start parallel 
    time_t start_parallel = clock();
    dim3 threadsPerBlock(16, 16);
    dim3 blocks(GRID_WIDTH/ threadsPerBlock.x + 1, GRID_HEIGHT / threadsPerBlock.y + 1);

    printf("initializing parallel grid...\n");
    init_array_d<<<blocks, threadsPerBlock>>>(-1);
    cudaDeviceSynchronize();
    simulate_parallel(); 

    int mem_size = sizeof(int) * GRID_HEIGHT * GRID_WIDTH;
    //get result form gpu
    cudaMemcpyFromSymbol(grid, grid_d, mem_size, 0, cudaMemcpyDeviceToHost);
    printf("%d", grid[100][500]);
    time_t end_parallel = clock();
    double elapsed_time_parallel  = (double)(end_parallel - start_parallel) / CLOCKS_PER_SEC;
    printf("parallel elapsed Time: %f seconds\n\n", elapsed_time_parallel);
    save_heat_map_to_binary_file(grid, "parallel_simulation.bin");

    //save simulation results
    return 0;
}