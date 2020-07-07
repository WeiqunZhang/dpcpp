#include <CL/sycl.hpp>

namespace sycl = cl::sycl;

#include <iostream>

//#define SUBGROUPSIZE 8   // works
//#define SUBGROUPSIZE 16  // works
#define SUBGROUPSIZE 32  // fail

int main (int argc, char* argv[])
{
    sycl::gpu_selector my_selector;
    sycl::device my_device(my_selector);
    sycl::context my_context(my_device);
    sycl::queue q(my_context, my_selector, sycl::property_list{sycl::property::queue::in_order{}});

    auto sgss = my_device.get_info<sycl::info::device::sub_group_sizes>();
    for (auto x : sgss) {
        std::cout << "Device sub group size: " << x << "\n";
    }

    int* p = (int*)sycl::malloc_device(SUBGROUPSIZE*sizeof(int), my_device, my_context);
    int* tot = (int*)sycl::malloc_device(sizeof(int), my_device, my_context);

    q.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(SUBGROUPSIZE),sycl::range<1>(SUBGROUPSIZE)),
        [=] (sycl::nd_item<1> item)
        {
            auto gid = item.get_global_id(0);
            p[gid] = gid;
        });
    });

    q.submit([&] (sycl::handler& h) {
        h.single_task([=] () { *tot = 0; });
    });

    q.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(SUBGROUPSIZE),sycl::range<1>(SUBGROUPSIZE)),
        [=] (sycl::nd_item<1> item)
        [[cl::intel_reqd_sub_group_size(SUBGROUPSIZE)]]
        {
            sycl::intel::sub_group const& sg = item.get_sub_group();
            auto gid = item.get_global_id(0);
            int x = p[gid];
            for (int offset = SUBGROUPSIZE/2; offset > 0; offset /= 2) {
                int y = sg.shuffle_down(x, offset);
                x += y;
            }
            if (gid == 0) {
                *tot = x;
            }
        });
    });

    q.wait();

    int htot;

    q.submit([&] (sycl::handler& h) { h.memcpy(&htot, tot, sizeof(int)); });
    q.wait();

    std::cout << "The total is " << htot << std::endl;

    sycl::free(p, my_context);
    sycl::free(tot, my_context);
}
