#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

void my_init_random (sycl::ordered_queue& q);
void my_finalize_random (sycl::ordered_queue& q);
SYCL_EXTERNAL int my_random ();
