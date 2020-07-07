#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

#include <iostream>
#include <cmath>
#include <cfenv>
#include <csignal>
#include <cstdlib>

void handler (int s)
{
    std::cout << "A signal " << s << " has been raised!" << std::endl;
    std::exit(EXIT_FAILURE);
}

int main (int argc, char* argv[])
{
    feenableexcept(FE_INVALID);  // trap floating point exceptions
    std::signal(SIGFPE, handler);

    sycl::gpu_selector my_selector;
    sycl::device my_device(my_selector);
    sycl::context my_context(my_device);
    sycl::queue q(my_context, my_selector, sycl::property_list{sycl::property::queue::in_order{}});

    const long n = 16;
    double* p = (double*)sycl::malloc_device(n*sizeof(double), my_device, my_context);

    q.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(n),sycl::range<1>(n)),
        [=] (sycl::nd_item<1> item)
        {
            auto gid = item.get_global_id(0);
            p[gid] = std::sin(gid);
        });        
    });
    q.wait();

    sycl::free(p, my_context);
}
