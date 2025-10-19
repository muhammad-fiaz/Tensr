/**
 * @file cpp_usage.cpp
 * @brief C++ usage examples for Tensr library
 * @author Muhammad Fiaz
 */

#include "tensr/tensr.hpp"
#include <iostream>

int main() {
    std::cout << "=== Tensr C++ Usage Examples ===\n\n";
    
    /* Create tensors */
    std::cout << "1. Creating tensors:\n";
    auto t1 = tensr::Tensor::zeros({2, 3});
    std::cout << "   Zeros tensor: ";
    t1.print();
    
    auto t2 = tensr::Tensor::ones({2, 3});
    std::cout << "   Ones tensor: ";
    t2.print();
    
    /* Arithmetic operations */
    std::cout << "\n2. Arithmetic operations:\n";
    auto t3 = t1 + t2;
    std::cout << "   zeros + ones = ";
    t3.print();
    
    auto t4 = t2 * t2;
    std::cout << "   ones * ones = ";
    t4.print();
    
    /* Range operations */
    std::cout << "\n3. Range operations:\n";
    auto t5 = tensr::Tensor::arange(0.0, 10.0, 2.0);
    std::cout << "   arange(0, 10, 2): ";
    t5.print();
    
    auto t6 = tensr::Tensor::linspace(0.0, 1.0, 5);
    std::cout << "   linspace(0, 1, 5): ";
    t6.print();
    
    /* Matrix operations */
    std::cout << "\n4. Matrix operations:\n";
    auto eye = tensr::Tensor::eye(3);
    std::cout << "   Identity matrix (3x3): ";
    eye.print();
    
    auto a = tensr::Tensor::full({2, 3}, 2.0);
    auto b = tensr::Tensor::full({3, 2}, 3.0);
    auto c = a.matmul(b);
    std::cout << "   Matrix multiplication result: ";
    c.print();
    
    /* Random tensors */
    std::cout << "\n5. Random tensors:\n";
    tensr::seed(42);
    auto rand_t = tensr::Tensor::rand({2, 3});
    std::cout << "   Random tensor: ";
    rand_t.print();
    
    /* Reduction operations */
    std::cout << "\n6. Reduction operations:\n";
    auto sum = t2.sum();
    std::cout << "   Sum of ones tensor: ";
    sum.print();
    
    auto mean = t2.mean();
    std::cout << "   Mean of ones tensor: ";
    mean.print();
    
    /* Mathematical functions */
    std::cout << "\n7. Mathematical functions:\n";
    auto t7 = tensr::Tensor::full({3}, 4.0);
    auto t8 = t7.sqrt();
    std::cout << "   sqrt(4.0): ";
    t8.print();
    
    auto t9 = tensr::Tensor::full({3}, 1.0);
    auto t10 = t9.exp();
    std::cout << "   exp(1.0): ";
    t10.print();
    
    std::cout << "\n=== Examples completed successfully! ===\n";
    return 0;
}
