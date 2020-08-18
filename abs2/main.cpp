#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

#include <iostream>

int main (int argc, char* argv[])
{
    sycl::gpu_selector my_selector;
    sycl::device my_device(my_selector);
    sycl::context my_context(my_device);
    sycl::queue q(my_context, my_selector, sycl::property_list{sycl::property::queue::in_order{}});

    double* p = (double*)sycl::malloc_device(2*sizeof(double), my_device, my_context);

    double x = -0.002;

    q.submit([&] (sycl::handler& h) {
        h.single_task([=] ()
        {
            *p = sycl::abs(x);
#if __SYCL_DEVICE_ONLY__
            static const __attribute__((opencl_constant)) char format[] = "sycl::abs(-0.002) = %f\n";
            cl::sycl::intel::experimental::printf(format, sycl::abs(x));
#endif
        });
    });
    q.wait();

    double hv;
    q.submit([&] (sycl::handler& h) { h.memcpy(&hv, p, sizeof(double)); });
    q.wait();

    std::cout << "sycl::abs on device: " << hv << "\n";

    sycl::free(p, my_context);
}
