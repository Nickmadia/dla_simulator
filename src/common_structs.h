// common.h

#ifndef COMMON_H
#define COMMON_H

#include <stdbool.h>
// Define your structs here
typedef struct {
    float x;
    float y;
    float horizontal_speed;
    float vertical_speed;
    bool solid;
} Particle;

#endif // COMMON_H
