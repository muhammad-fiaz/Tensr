/**
 * @file random.c
 * @brief Random number generation for tensors
 * @author Muhammad Fiaz
 * 
 * Provides random number generation functions for creating tensors with
 * uniform and normal distributions, similar to numpy.random functionality.
 */

#include "tensr/tensr.h"
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static unsigned int rng_state = 0;

/**
 * @brief Set the random seed for reproducible random number generation
 * @param seed Unsigned integer seed value
 * 
 * Sets the seed for the random number generator. Use the same seed to get
 * reproducible results across runs. Similar to numpy.random.seed().
 */
void tensr_seed(unsigned int seed) {
    rng_state = seed;
    srand(seed);
}

/**
 * @brief Generate a random number from uniform distribution [0, 1)
 * @return Random double in range [0, 1)
 */
static double rand_uniform() {
    return (double)rand() / (double)RAND_MAX;
}

/**
 * @brief Generate a random number from standard normal distribution
 * @return Random double from N(0, 1) distribution
 * 
 * Uses Box-Muller transform to generate normally distributed random numbers.
 */
static double rand_normal() {
    double u1 = rand_uniform();
    double u2 = rand_uniform();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

/**
 * @brief Create a tensor with random values from uniform distribution [0, 1)
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param device Device to create tensor on (CPU, CUDA, etc.)
 * @return Pointer to newly created tensor with random values
 * 
 * Creates a tensor filled with random values from a uniform distribution
 * in the range [0, 1). Similar to numpy.random.rand().
 * 
 * Example:
 *   size_t shape[] = {3, 3};
 *   Tensor* t = tensr_rand(shape, 2, TENSR_CPU);
 */
Tensor* tensr_rand(size_t* shape, size_t ndim, TensrDevice device) {
    if (rng_state == 0) tensr_seed((unsigned int)time(NULL));

    Tensor* t = tensr_create(shape, ndim, TENSR_FLOAT32, device);
    if (!t) return NULL;

    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        data[i] = (float)rand_uniform();
    }
    return t;
}

/**
 * @brief Create a tensor with random values from standard normal distribution
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param device Device to create tensor on (CPU, CUDA, etc.)
 * @return Pointer to newly created tensor with random values
 * 
 * Creates a tensor filled with random values from a standard normal
 * distribution (mean=0, std=1). Similar to numpy.random.randn().
 * 
 * Example:
 *   size_t shape[] = {3, 3};
 *   Tensor* t = tensr_randn(shape, 2, TENSR_CPU);
 */
Tensor* tensr_randn(size_t* shape, size_t ndim, TensrDevice device) {
    if (rng_state == 0) tensr_seed((unsigned int)time(NULL));

    Tensor* t = tensr_create(shape, ndim, TENSR_FLOAT32, device);
    if (!t) return NULL;

    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        data[i] = (float)rand_normal();
    }
    return t;
}

/**
 * @brief Create a tensor with random integers in range [low, high)
 * @param low Lower bound (inclusive)
 * @param high Upper bound (exclusive)
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param device Device to create tensor on (CPU, CUDA, etc.)
 * @return Pointer to newly created tensor with random integers
 * 
 * Creates a tensor filled with random integers from the discrete uniform
 * distribution in the range [low, high). Similar to numpy.random.randint().
 * 
 * Example:
 *   size_t shape[] = {3, 3};
 *   Tensor* t = tensr_randint(0, 10, shape, 2, TENSR_CPU);
 */
Tensor* tensr_randint(int low, int high, size_t* shape, size_t ndim, TensrDevice device) {
    if (rng_state == 0) tensr_seed((unsigned int)time(NULL));

    Tensor* t = tensr_create(shape, ndim, TENSR_INT32, device);
    if (!t) return NULL;

    int32_t* data = (int32_t*)t->data;
    int range = high - low;
    for (size_t i = 0; i < t->size; i++) {
        data[i] = low + (rand() % range);
    }
    return t;
}
