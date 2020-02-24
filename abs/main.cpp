#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include <iostream>

int main (int argc, char* argv[])
{
    std::cout << " sycl::abs(-6) = " <<  sycl::abs(-6) << "\n"
              << "-sycl::abs(-6) = " << -sycl::abs(-6) << "\n";
}
