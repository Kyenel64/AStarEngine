#pragma once

#include <cmath>
#include <limits>
#include <memory>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Utility Functions

__device__ inline float degrees_to_radians(float degrees) {
    return float(degrees * 3.1415926535897932385 / 180.0);
}

// Common Headers

#include "ray.cuh"
#include "vec3.cuh"