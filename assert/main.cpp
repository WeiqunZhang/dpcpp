#include <CL/sycl.hpp>
#include <iostream>
#include <cassert>
#include <cstdlib>
namespace sycl = cl::sycl;

namespace {
    auto error_handler = [] (sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch (sycl::exception const& ex) {
                std::cerr << "Async SYCL exception: " << ex.what() << "!" << std::endl;
                std::abort();
            }
        }
    };
}

int main (int argc, char* argv[])
{
    {
        sycl::queue q(sycl::gpu_selector{}, error_handler);
        q.submit([&] (sycl::handler& h) {
            h.single_task([=] () {
                assert(0);
            });
        });
    }

    std::cout << "If you see this message, assert(0) did not abort or throw an error." << std::endl;
}
