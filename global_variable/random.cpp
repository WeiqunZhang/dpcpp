#include "random.H"

namespace {
    int* state = nullptr;
    __attribute__((opencl_global)) int* d_state;
    // CUDA: __device__ int* d_state;
}

void my_init_random (sycl::ordered_queue& q)
{
    state = (int*)sycl::malloc_device(sizeof(int),
                                      q.get_device(), q.get_context());
    auto p = state;
    q.submit([&] (sycl::handler& h) {
        h.single_task([=] () {
            *p = 0;
            // CUDA: d_state = p; or use cudaMemcpyToSymbol
        });
    });
}

void my_finalize_random (sycl::ordered_queue& q)
{
    free(state, q.get_context());
}

int my_random ()
{
    // error: SYCL kernel cannot use a global variable
    // return (*d_state)++;
    return 0;
}
