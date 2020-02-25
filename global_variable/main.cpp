#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#include "random.H"

int main (int argc, char* argv[])
{
    sycl::ordered_queue q(sycl::gpu_selector{});

    my_init_random(q);

    q.submit([&] (sycl::handler& h) {
        h.single_task([=] () {
            auto r = my_random();
#ifdef __SYCL_DEVICE_ONLY__
            static const __attribute__((opencl_constant)) char fmt[] = "%d\n";
            sycl::intel::experimental::printf(fmt, r);
#endif
        });
    });

    q.submit([&] (sycl::handler& h) {
        h.single_task([=] () {
            auto r = my_random();
#ifdef __SYCL_DEVICE_ONLY__
            static const __attribute__((opencl_constant)) char fmt[] = "%d\n";
            sycl::intel::experimental::printf(fmt, r);
#endif
        });
    });

    q.wait();

    my_finalize_random(q);
}
