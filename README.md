# Diffusion-Limited Aggregation (DLA) Project

This repository contains the source code for a Diffusion-Limited Aggregation (DLA) simulation implemented in both serial and parallel (CUDA) versions. DLA is a fascinating phenomenon in physics and chemistry where particles undergo random motion and aggregate to form complex structures.
![](https://raw.githubusercontent.com/Nickmadia/dla_simulator/master/examples/heatmap_parallel.png)
## Overview

The project consists of two main implementations:

1. **Serial Implementation**: This version simulates DLA using a single-threaded approach. The particles move in a 2D grid, and their motion is influenced by random walks until they aggregate into clusters.

2. **Parallel (CUDA) Implementation**: The parallel version leverages CUDA, a parallel computing platform, to speed up the simulation. Multiple threads work concurrently to simulate particle motion, taking advantage of GPU parallelism.

3. **Parallel (OMP) Implementation**: Here I am using the OMP API in order to get a significant speedup in the simulation.
   
## Results and Visualization
Explore the `examples/` directory for visualizations and images generated from the simulations.
