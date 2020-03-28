#include <iostream>
#include <chrono>

__device__
double my_atomic_add (double* address, double val)
{
    using R = double;
    using I = unsigned long long;
    static_assert(sizeof(R) == sizeof(I), "sizeof R != sizeof I");
    I* add_as_I = reinterpret_cast<I*>(address);
    I old_I = *add_as_I, assumed_I;
    do {
        assumed_I = old_I;
        R const new_R = *(reinterpret_cast<R const*>(&assumed_I)) + val;
        old_I = atomicCAS(add_as_I, assumed_I, *(reinterpret_cast<I const*>(&new_R)));
    } while (assumed_I != old_I);
    return *(reinterpret_cast<R const*>(&old_I));
}

template <class L> __global__ void launch_global (L f) { f(); }

int main (int argc, char* argv[])
{
    const long n = 128L*128L*128L;
    double* p;
    cudaMalloc(&p, n*sizeof(double));

    launch_global<<<n/256, 256>>>([=] __device__ () noexcept {
        int tid = blockDim.x*blockIdx.x + threadIdx.x;
        p[tid] = tid;
    });
    cudaDeviceSynchronize();

    double h_sum = 0.0;
    double* d_sum;
    cudaMalloc(&d_sum, sizeof(double));
    cudaMemcpy(d_sum, &h_sum, sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // warming up
    launch_global<<<n/256, 256>>>([=] __device__ () noexcept {
        my_atomic_add(d_sum, p[blockDim.x*blockIdx.x + threadIdx.x]);
    });
    cudaDeviceSynchronize();

    auto t0 = std::chrono::high_resolution_clock::now();
    launch_global<<<n/256, 256>>>([=] __device__ () noexcept {
        my_atomic_add(d_sum, p[blockDim.x*blockIdx.x + threadIdx.x]);
    });
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Time in my atomic add: " << std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0).count() << std::endl;;

    cudaMemcpy(d_sum, &h_sum, sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    auto t2 = std::chrono::high_resolution_clock::now();
    launch_global<<<n/256, 256>>>([=] __device__ () noexcept {
        atomicAdd(d_sum, p[blockDim.x*blockIdx.x + threadIdx.x]);
    });
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Time in CUDA atomicAdd: " << std::chrono::duration_cast<std::chrono::duration<double>>(t3-t2).count() << std::endl;;

    cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout.precision(17);
    std::cout << "sum = " << h_sum << ", " << n*(n-1)/2 << std::endl;

    cudaFree(d_sum);
    cudaFree(p);
}
