#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#define GROUP_SIZE 128
#define TOTAL_SIZE 128

// Assume this function is in a header file included here.
inline void transpose (int* p)
{
#if 0
    // This is what CUDA could do.
    __shared__ int shared_data[GROUP_SIZE];
    int id = threadIdx.x;
    shared_data[id] = p[id];
    __syncthreads();
    p[id] = shared_data[GROUP_SIZE-1-id];
#endif    
}

int main (int argc, char* argv[])
{
    sycl::ordered_queue q(sycl::gpu_selector{});

    int* p = (int*)sycl::malloc_device(sizeof(int)*TOTAL_SIZE,
                                       q.get_device(), q.get_context());

    q.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(TOTAL_SIZE),
                                         sycl::range<1>(GROUP_SIZE)),
        [=] (sycl::nd_item<1> item)
        {
            int id = item.get_global_id(0);
            p[id] = id;
        });
    });

    q.submit([&] (sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(TOTAL_SIZE),
                                         sycl::range<1>(GROUP_SIZE)),
        [=] (sycl::nd_item<1> item)
        {
            transpose(p);
        });
    });

    q.wait();
    free(p, q.get_context());
}
