#include "dla_parallel.h"

__device__ __host__ void make_static(Particle *particle, int *grid, int tick) {
    //TODO implement circle check instead of squared
    for ( int i = -PARTICLE_RADIUS; i <= PARTICLE_RADIUS; i++) {
        for (int j = -PARTICLE_RADIUS; j<= PARTICLE_RADIUS; j++) {
            int dx = (int)particle->x - i;
            int dy = (int)particle->y - j;
            // if in bound
            float da = (float) particle-> x - dx;
            float db = (float) particle-> y - dy;
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
__device__ __host__ void check_hit(Particle *particle, int * grid, int tick) {
    bool loop = true;
    for ( int i = -PARTICLE_RADIUS; i <= PARTICLE_RADIUS && loop; i++) {
        for (int j = -PARTICLE_RADIUS; j<= PARTICLE_RADIUS; j++) {
            int dx = particle->x - i;
            int dy = particle->y - j;
            // if in bound
            if (dx >= 0 && dx < GRID_WIDTH && dy >= 0 && dy < GRID_HEIGHT) {
                float da = (float) particle-> x - dx;
                float db = (float) particle-> y - dy;
                float d = sqrt((float)(i*i) + (float)(j *j));
                if(d<= PARTICLE_RADIUS) {
                if (grid[dy* GRID_WIDTH + dx] >= 0) {
                    make_static(particle, grid, tick);
                    loop = false;
                    break;
                }
                }
            }
        }
    }
}
__global__ void init_rng_states_d(curandState *states, unsigned long long seed) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid<PARTICLE_COUNT) {
        curand_init(seed, tid, 0, &states[tid]);
    }
   
}

__global__ void init_array_d(int *grid_d, int value) {
    //make reading it easier
    grid_d[(blockIdx.y * blockDim.y + threadIdx.y) * GRID_WIDTH + (blockIdx.x * blockDim.x + threadIdx.x)] = value;
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
__global__ void init_seeds( int *grid_d, curandState *state) {
    if (SEED_POSITION == "center") {
        grid_d[(GRID_HEIGHT* GRID_WIDTH)/2 + (GRID_WIDTH/2)] = 1;
    } else if ( SEED_POSITION == "random") {
       for (int i =0; i< SEED_COUNT; i++) {
          int x = (int)random_float_d(state, 0,GRID_HEIGHT);
          int y =(int) random_float_d(state, 0,GRID_WIDTH);
          grid_d[y * GRID_WIDTH + x] = 1;
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

__global__ void tick( int * grid_d, Particle *particles, curandState * states, int k) {

    int i = threadIdx.x + blockIdx.x * blockDim.x; 
    Particle *particle = particles + i;
    if (i < PARTICLE_COUNT) {
        if (!particle->solid) {
            move_particle_d(particle, states);
            check_hit(particle,grid_d,k);
        }
    }
}
int * init_grid() {
    int * grid_d; 
    int array_size = GRID_WIDTH * GRID_HEIGHT * sizeof(int);

    // allocating grid on the GPU
    cudaMalloc((void**)&grid_d, array_size);
    dim3 threadsPerBlock(16, 16);
    dim3 blocks(GRID_WIDTH/ threadsPerBlock.x + 1, GRID_HEIGHT / threadsPerBlock.y + 1);

    printf("initializing parallel grid...\n");
    //init grid to -1
    init_array_d<<<blocks, threadsPerBlock>>>(grid_d,-1);
    cudaDeviceSynchronize();
    return grid_d;
}
curandState* init_rng_states( int threads, int blocks) {

    curandState *d_rngStates;
    cudaMalloc((void**)&d_rngStates, PARTICLE_COUNT * sizeof(curandState));
    
    int seed = rand();
    init_rng_states_d<<<blocks,threads>>>(d_rngStates, seed);
    cudaDeviceSynchronize();
    return d_rngStates;
}
Particle * init_particles( curandState * d_rngStates, int threads, int blocks) {
    Particle * particles_d;
    size_t mem_size = PARTICLE_COUNT * sizeof(Particle);
    cudaMalloc(&particles_d, mem_size);
    // init particles parallel
    init_particles_d<<<blocks, threads>>>(particles_d, d_rngStates);
    cudaDeviceSynchronize();
    return particles_d;

}
void check_cuda_error() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "initRNGStates launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}
void simulate_parallel(int * grid) {

    int *grid_d = init_grid();
    

    //init random generator
    int threads_block = 16;
    int sim_blocks = (PARTICLE_COUNT)  / threads_block ;

    curandState* d_rngStates = init_rng_states( threads_block, sim_blocks);
    check_cuda_error();
   
    //init seeds 
    init_seeds<<<1,1>>>( grid_d, d_rngStates);
    cudaDeviceSynchronize();
    check_cuda_error();

    //init particles array
    
    Particle* particles_d = init_particles( d_rngStates, threads_block,sim_blocks);

    check_cuda_error();
    printf("starting parallel simulation...\n\n");
    
    for( int i = 0; i< ITERATIONS; i++) {
        tick<<<sim_blocks,threads_block>>>( grid_d, particles_d, d_rngStates, i);
        cudaDeviceSynchronize();
    }

    check_cuda_error();
    //get result form gpu
    cudaMemcpy(grid, grid_d, GRID_HEIGHT * GRID_WIDTH * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(grid_d);
    cudaFree(d_rngStates);
    cudaFree(particles_d);
}