#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

#include <iostream>
#include <cmath>

int main (int argc, char* argv[])
{
    sycl::gpu_selector my_selector;
    sycl::device my_device(my_selector);
    sycl::context my_context(my_device);
    sycl::queue q(my_context, my_selector, sycl::property_list{sycl::property::queue::in_order{}});

    double* p = (double*)sycl::malloc_device(2*sizeof(double), my_device, my_context);

    double x = -0.002;
    double y =  0.002;

    q.submit([&] (sycl::handler& h) {
        h.single_task([=] ()
        {
            p[0] = sycl::hypot(x,y);
            p[1] = sycl::hypot(-0.002,0.002);
        });
    });
    q.wait();

    double hv[2];
    q.submit([&] (sycl::handler& h) { h.memcpy(&hv, p, sizeof(double)*2); });
    q.wait();

    std::cout << "sycl::hypot on device: " << hv[0] << ", " << hv[1] << "\n"
              << "std::hypot on host: " << std::hypot(x,y) << "\n";

    sycl::free(p, my_context);
}
