#ifndef DLA_PARALLELL
#define DLA_PARALLELL
#include "dla_parallel.h"

__global__ void initRNGStates(curandState *states, unsigned long long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<PARTICLE_COUNT) {
        curand_init(seed, tid, 0, &states[tid]);
    }
   
}

__global__ void init_array_d(int (*grid_d)[GRID_WIDTH], int value) {
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
__global__ void init_seeds(int (*grid_d)[GRID_WIDTH], curandState *state) {
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


__global__ void tick(int (*grid_d)[GRID_WIDTH], Particle *particles, curandState * states, int k) {

    int i = threadIdx.x + blockIdx.x * blockDim.x; 
    Particle *particle = particles + i;
    if (i < PARTICLE_COUNT) {
        if (!particle->solid) {
            move_particle_d(particle, states);
            check_hit(particle,grid_d,k);
        }
    }
}
void simulate_parallel(int (*grid_d)[GRID_WIDTH]) {
    int threads_block = 16;
    int sim_blocks = (PARTICLE_COUNT)  / threads_block ;
    //init random generator
    curandState *d_rngStates;
    cudaMalloc((void**)&d_rngStates, PARTICLE_COUNT * sizeof(curandState));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cudamalloc failed: %s\n", cudaGetErrorString(err));
        // Additional error handling or exit if necessary
    }
    int seed = rand();
    initRNGStates<<<sim_blocks,threads_block>>>(d_rngStates,seed);
    cudaDeviceSynchronize();
    
    init_seeds<<<1,1>>>(grid_d, d_rngStates);

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
        tick<<<sim_blocks,threads_block>>>(grid_d, particles_d, d_rngStates, i);
        cudaDeviceSynchronize();
    }
    cudaFree(d_rngStates);
    cudaFree(particles_d);
}
#endif //DLA_PARALLELL