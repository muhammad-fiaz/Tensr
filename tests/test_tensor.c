/**
 * @file test_tensor.c
 * @brief Test suite for Tensr library
 * @author Muhammad Fiaz
 */

#include "tensr/tensr.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>

void test_create() {
    printf("Testing tensor creation...\n");
    size_t shape[] = {2, 3};
    Tensor* t = tensr_create(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    assert(t != NULL);
    assert(t->ndim == 2);
    assert(t->shape[0] == 2);
    assert(t->shape[1] == 3);
    assert(t->size == 6);
    tensr_free(t);
    printf("✓ Tensor creation test passed\n");
}

void test_zeros() {
    printf("Testing zeros...\n");
    size_t shape[] = {3, 3};
    Tensor* t = tensr_zeros(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    assert(t != NULL);
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        assert(data[i] == 0.0f);
    }
    tensr_free(t);
    printf("✓ Zeros test passed\n");
}

void test_ones() {
    printf("Testing ones...\n");
    size_t shape[] = {2, 2};
    Tensor* t = tensr_ones(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    assert(t != NULL);
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        assert(data[i] == 1.0f);
    }
    tensr_free(t);
    printf("✓ Ones test passed\n");
}

void test_arange() {
    printf("Testing arange...\n");
    Tensor* t = tensr_arange(0.0, 10.0, 1.0, TENSR_FLOAT32, TENSR_CPU);
    assert(t != NULL);
    assert(t->size == 10);
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        assert(fabs(data[i] - (float)i) < 1e-6);
    }
    tensr_free(t);
    printf("✓ Arange test passed\n");
}

void test_eye() {
    printf("Testing eye...\n");
    Tensor* t = tensr_eye(3, TENSR_FLOAT32, TENSR_CPU);
    assert(t != NULL);
    assert(t->shape[0] == 3);
    assert(t->shape[1] == 3);
    float* data = (float*)t->data;
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            if (i == j) {
                assert(data[i * 3 + j] == 1.0f);
            } else {
                assert(data[i * 3 + j] == 0.0f);
            }
        }
    }
    tensr_free(t);
    printf("✓ Eye test passed\n");
}

void test_arithmetic() {
    printf("Testing arithmetic operations...\n");
    size_t shape[] = {2, 2};
    Tensor* a = tensr_ones(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    Tensor* b = tensr_full(shape, 2, 2.0, TENSR_FLOAT32, TENSR_CPU);
    
    Tensor* c = tensr_add(a, b);
    assert(c != NULL);
    float* data = (float*)c->data;
    for (size_t i = 0; i < c->size; i++) {
        assert(fabs(data[i] - 3.0f) < 1e-6);
    }
    
    Tensor* d = tensr_mul(a, b);
    float* data_d = (float*)d->data;
    for (size_t i = 0; i < d->size; i++) {
        assert(fabs(data_d[i] - 2.0f) < 1e-6);
    }
    
    tensr_free(a);
    tensr_free(b);
    tensr_free(c);
    tensr_free(d);
    printf("✓ Arithmetic operations test passed\n");
}

void test_reduction() {
    printf("Testing reduction operations...\n");
    size_t shape[] = {2, 3};
    Tensor* t = tensr_ones(shape, 2, TENSR_FLOAT32, TENSR_CPU);
    
    Tensor* sum = tensr_sum(t, NULL, 0, false);
    assert(sum != NULL);
    float* sum_data = (float*)sum->data;
    assert(fabs(sum_data[0] - 6.0f) < 1e-6);
    
    Tensor* mean = tensr_mean(t, NULL, 0, false);
    float* mean_data = (float*)mean->data;
    assert(fabs(mean_data[0] - 1.0f) < 1e-6);
    
    tensr_free(t);
    tensr_free(sum);
    tensr_free(mean);
    printf("✓ Reduction operations test passed\n");
}

void test_matmul() {
    printf("Testing matrix multiplication...\n");
    size_t shape_a[] = {2, 3};
    size_t shape_b[] = {3, 2};
    
    Tensor* a = tensr_ones(shape_a, 2, TENSR_FLOAT32, TENSR_CPU);
    Tensor* b = tensr_ones(shape_b, 2, TENSR_FLOAT32, TENSR_CPU);
    
    Tensor* c = tensr_matmul(a, b);
    assert(c != NULL);
    assert(c->shape[0] == 2);
    assert(c->shape[1] == 2);
    
    float* data = (float*)c->data;
    for (size_t i = 0; i < c->size; i++) {
        assert(fabs(data[i] - 3.0f) < 1e-6);
    }
    
    tensr_free(a);
    tensr_free(b);
    tensr_free(c);
    printf("✓ Matrix multiplication test passed\n");
}

void test_random() {
    printf("Testing random operations...\n");
    size_t shape[] = {10, 10};
    
    tensr_seed(42);
    Tensor* t = tensr_rand(shape, 2, TENSR_CPU);
    assert(t != NULL);
    assert(t->size == 100);
    
    float* data = (float*)t->data;
    for (size_t i = 0; i < t->size; i++) {
        assert(data[i] >= 0.0f && data[i] <= 1.0f);
    }
    
    tensr_free(t);
    printf("✓ Random operations test passed\n");
}

void test_io() {
    printf("Testing I/O operations...\n");
    size_t shape[] = {2, 3};
    Tensor* t = tensr_arange(0.0, 6.0, 1.0, TENSR_FLOAT32, TENSR_CPU);
    t->shape[0] = 2;
    t->shape[1] = 3;
    t->ndim = 2;
    
    int result = tensr_save("test_tensor.bin", t);
    assert(result == 0);
    
    Tensor* loaded = tensr_load("test_tensor.bin");
    assert(loaded != NULL);
    assert(loaded->ndim == t->ndim);
    assert(loaded->size == t->size);
    
    tensr_free(t);
    tensr_free(loaded);
    printf("✓ I/O operations test passed\n");
}

int main() {
    printf("=== Tensr Library Test Suite ===\n\n");
    
    test_create();
    test_zeros();
    test_ones();
    test_arange();
    test_eye();
    test_arithmetic();
    test_reduction();
    test_matmul();
    test_random();
    test_io();
    
    printf("\n=== All tests passed! ===\n");
    return 0;
}
