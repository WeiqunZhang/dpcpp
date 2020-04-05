#include <CL/sycl.hpp>
#include <iostream>
#include <cassert>
#include <cstdlib>
namespace sycl = cl::sycl;

#undef NDEBUG

int main (int argc, char* argv[])
{
    {
        sycl::queue q(sycl::gpu_selector{});
        q.submit([&] (sycl::handler& h) {
            h.single_task([=] () {
                assert(0);
            });
        });
    }

    std::cout << "If you see this message, assert(0) did not abort." << std::endl;
}
