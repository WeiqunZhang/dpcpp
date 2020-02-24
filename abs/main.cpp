#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include <iostream>
#include <type_traits>

int main (int argc, char* argv[])
{
    std::cout << " sycl::abs(-6) = " <<  sycl::abs(-6) << "\n"
              << "-sycl::abs(-6) = " << -sycl::abs(-6) << "\n";
    auto tsycl = sycl::abs(-6);
    auto tstd  =  std::abs(-6);
    std::cout << "Is sycl::abs(-6) signed? " << std::is_signed<decltype(tsycl)>::value << "\n"
              << "Is  std::abs(-6) signed? " << std::is_signed<decltype(tstd )>::value << "\n";
}
