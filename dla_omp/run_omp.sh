#!/bin/bash
make
./bin/dla_omp
cd ../visualizer

cargo run