#include <CL/sycl.hpp>
#include <iostream>
#include <chrono>

namespace sycl = cl::sycl;

double my_atomic_add (double* address, double val)
{
    using R = double;
    using I = unsigned long long;
    constexpr auto mo = sycl::memory_order::relaxed;
    constexpr auto as = sycl::access::address_space::global_space;
    static_assert(sizeof(R) == sizeof(I), "sizeof R != sizeof I");
    I* add_as_I = reinterpret_cast<I*>(address);
    sycl::atomic<I,as> a{sycl::multi_ptr<I,as>(add_as_I)};
    I old_I = a.load(mo), new_I;
    do {
        R const new_R = *(reinterpret_cast<R const*>(&old_I)) + val;
        new_I = *(reinterpret_cast<I const*>(&new_R));
    } while (not a.compare_exchange_strong(old_I, new_I, mo));
    return *(reinterpret_cast<R const*>(&old_I));
}

int main (int argc, char* argv[])
{
    sycl::gpu_selector my_selector;
    sycl::device my_device(my_selector);
    sycl::context my_context(my_device);
    sycl::ordered_queue q(my_context, my_selector);

    const long n = 128L*128L*128L;
    double* p = (double*)sycl::malloc_device(n*sizeof(double), my_device, my_context);

    auto t0 = std::chrono::high_resolution_clock::now();
    q.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(n),sycl::range<1>(256)),
        [=] (sycl::nd_item<1> item)
        {
            auto gid = item.get_global_id(0);
            p[gid] = gid;
        });        
    });
    q.wait();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0).count() << std::endl;;

    double h_sum = 0.0;
    double* d_sum = (double*)sycl::malloc_device(sizeof(double), my_device, my_context);
    q.submit([&] (sycl::handler& h) { h.memcpy(d_sum, &h_sum, sizeof(double)); });
    q.wait();

    q.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(n),sycl::range<1>(256)),
        [=] (sycl::nd_item<1> item)
        {
            my_atomic_add(d_sum, p[item.get_global_id(0)]);
        });
    });
    q.wait();

    q.submit([&] (sycl::handler& h) { h.memcpy(&h_sum, d_sum, sizeof(double)); });
    q.wait();
    std::cout.precision(17);
    std::cout << "sum = " << h_sum << ", " << n*(n-1)/2 << std::endl;

    sycl::free(d_sum, my_context);
    sycl::free(p,     my_context);
}
