// common.h

#ifndef COMMON_H
#define COMMON_H

#include <stdbool.h>
// Define your structs here
typedef struct {
    int x;
    int y;
    bool solid;
} Particle;
typedef struct {
    float x;
    float y;
} Coord;
#endif // COMMON_H
