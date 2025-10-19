/**
 * @file basic_usage.c
 * @brief Basic usage examples for Tensr library
 * @author Muhammad Fiaz
 */

#include "tensr/tensr.h"
#include <stdio.h>

int main() {
    printf("=== Tensr Basic Usage Examples ===\n\n");
    
    /* Create a 2x3 tensor */
    printf("1. Creating tensors:\n");
    size_t shape[] = {2, 3};
    Tensor* t1 = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    printf("   Zeros tensor: ");
    tensr_print(t1);
    
    Tensor* t2 = tensr_ones(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    printf("   Ones tensor: ");
    tensr_print(t2);
    
    /* Arithmetic operations */
    printf("\n2. Arithmetic operations:\n");
    Tensor* t3 = tensr_add(t1, t2);
    printf("   zeros + ones = ");
    tensr_print(t3);
    
    Tensor* t4 = tensr_mul(t2, t2);
    printf("   ones * ones = ");
    tensr_print(t4);
    
    /* Arange and linspace */
    printf("\n3. Range operations:\n");
    Tensor* t5 = tensr_arange(0.0, 10.0, 2.0, TENSR_FLOAT32, TENSR_CPU);
    printf("   arange(0, 10, 2): ");
    tensr_print(t5);
    
    Tensor* t6 = tensr_linspace(0.0, 1.0, 5, TENSR_FLOAT32, TENSR_CPU);
    printf("   linspace(0, 1, 5): ");
    tensr_print(t6);
    
    /* Matrix operations */
    printf("\n4. Matrix operations:\n");
    Tensor* eye = tensr_eye(3, TENSR_FLOAT32, TENSR_CPU);
    printf("   Identity matrix (3x3): ");
    tensr_print(eye);
    
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3, 2};
    Tensor* a = tensr_full(shape_a, 2, 2.0, TENSR_FLOAT32, TENSR_CPU);
    Tensor* b = tensr_full(shape_b, 2, 3.0, TENSR_FLOAT32, TENSR_CPU);
    Tensor* c = tensr_matmul(a, b);
    printf("   Matrix multiplication result: ");
    tensr_print(c);
    
    /* Random tensors */
    printf("\n5. Random tensors:\n");
    tensr_seed(42);
    size_t rand_shape[] = {2, 3};
    Tensor* rand_t = tensr_rand(rand_shape, 2, TENSR_CPU);
    printf("   Random tensor: ");
    tensr_print(rand_t);
    
    /* Reduction operations */
    printf("\n6. Reduction operations:\n");
    Tensor* sum = tensr_sum(t2, NULL, 0, false);
    printf("   Sum of ones tensor: ");
    tensr_print(sum);
    
    Tensor* mean = tensr_mean(t2, NULL, 0, false);
    printf("   Mean of ones tensor: ");
    tensr_print(mean);
    
    /* Cleanup */
    tensr_free(t1);
    tensr_free(t2);
    tensr_free(t3);
    tensr_free(t4);
    tensr_free(t5);
    tensr_free(t6);
    tensr_free(eye);
    tensr_free(a);
    tensr_free(b);
    tensr_free(c);
    tensr_free(rand_t);
    tensr_free(sum);
    tensr_free(mean);
    
    printf("\n=== Examples completed successfully! ===\n");
    return 0;
}
