#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

int main (int argc, char* argv[])
{
    {
        sycl::gpu_selector my_selector;
        sycl::device my_device(my_selector);
        sycl::context my_context(my_device);
        sycl::ordered_queue q(my_context, my_selector);
        try {
            q.submit([&] (sycl::handler& h) {
                h.parallel_for(sycl::nd_range<1>(sycl::range<1>(8),sycl::range<1>(8)),
                [=] (sycl::nd_item<1> item)
                {
                    //
                });
                });
        } catch (sycl::exception const& ex) {
            std::cout << "sycl::exception caught: " << ex.what() << std::endl;
        }
    }
    std::cout << "End of program\n";
}
