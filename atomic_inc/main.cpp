#include <CL/sycl.hpp>
#include <iostream>

namespace sycl = cl::sycl;

template <sycl::access::address_space AS = sycl::access::address_space::global_space>
unsigned int my_atomicInc (unsigned int* const m, unsigned int const value)
{
    constexpr auto mo = sycl::memory_order::relaxed;
    sycl::atomic<unsigned int,AS> a{sycl::multi_ptr<unsigned int,AS>(m)};
    unsigned int oldi = a.load(mo), newi;
    do {
        newi = (oldi >= value) ? 0u : (oldi+1u);
    } while (not a.compare_exchange_strong(oldi, newi, mo));
    return oldi;
}

int main (int argc, char* argv[])
{
    sycl::gpu_selector my_selector;
    sycl::device my_device(my_selector);
    sycl::context my_context(my_device);
    sycl::queue q(my_context, my_device, sycl::property_list{sycl::property::queue::in_order{}});

    unsigned int* p = (unsigned int*)sycl::malloc_device(sizeof(unsigned int),
                                                         my_device, my_context);
    unsigned int* id = (unsigned int*)sycl::malloc_device(sizeof(unsigned int)*2,
                                                          my_device, my_context);

    q.submit([&] (sycl::handler& h)
    {
        h.single_task([=] ()
        {
            *p = 0;
            id[0] = 999;
            id[1] = 999;
        });
    });

    q.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(256),
                                         sycl::range<1>(128)),
        [=] (sycl::nd_item<1> item)
        {
            auto tid = item.get_local_id(0);
            auto bid = item.get_group(0);
            if (tid == 0) {
                auto vid = my_atomicInc<sycl::access::address_space::global_space>(p,2);
                id[bid] = vid;
            }
        });        
    });
    q.wait();

    unsigned int hid[2];
    q.submit([&] (sycl::handler& h) { h.memcpy(hid, id, sizeof(unsigned int)*2); });
    q.wait();

    sycl::free(p, my_context);
    sycl::free(id, my_context);

    std::cout << "Virtual group ids: " << hid[0] << ", " << hid[1] << std::endl;
}
