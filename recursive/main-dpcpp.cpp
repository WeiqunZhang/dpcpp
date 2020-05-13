#include <CL/sycl.hpp>
#include <cmath>
namespace sycl = cl::sycl;

int main (int argc, char* argv[])
{
    {
        sycl::gpu_selector my_selector;
        sycl::device my_device(my_selector);
        sycl::context my_context(my_device);
        sycl::queue q(my_context, my_selector);
        double* p = (double*)sycl::malloc_device(sizeof(double), my_device, my_context);
        q.submit([&] (sycl::handler& h) {
            h.single_task([=] () {
                *p = std::sin(3.14);  // This does not work with AOT.
                // *p = sycl::sin(3.14);  // This works with AOT.
            });
        });
        q.wait();
        sycl::free(p, my_context);
    }
}
