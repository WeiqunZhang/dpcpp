#include "GpuParser.H"
#include <iostream>
#include <cstdlib>
#include <random>
#include <vector>
#include <cmath>

template<class L> AMREX_GPU_GLOBAL void launch (L f0) { f0(); }

void* myMalloc (std::size_t s)
{
    void*p;
    cudaMallocManaged(&p, s);
    return p;
}

void myFree (void* p)
{
    cudaFree(p);
}

int main (int argc, char* argv[])
{
    constexpr int N = 256;

    WarpXParser cpu_parser("sqrt((x**2+y**2+z**2)/(3.*3.))*5.");
    cpu_parser.registerVariables({"x","y","z"});

    std::vector<double> h_x(N);
    std::vector<double> h_y(N);
    std::vector<double> h_z(N);
    std::vector<double> h_r(N);
    {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        constexpr double pi = 3.14159265358979323846264338327950288;
        for (int i = 0; i < N; ++i) {
            double theta = pi * distribution(generator);
            double phi = 2.*pi*distribution(generator);
            h_x[i] = 3.*std::sin(theta)*std::cos(phi);
            h_y[i] = 3.*std::sin(theta)*std::sin(phi);
            h_z[i] = 3.*std::cos(theta);
        }
    }

    cudaSetDevice(0);

    double *d_x = (double*)myMalloc(sizeof(double)*N);
    double *d_y = (double*)myMalloc(sizeof(double)*N);
    double *d_z = (double*)myMalloc(sizeof(double)*N);
    double *d_r = (double*)myMalloc(sizeof(double)*N);

    cudaMemcpy(d_x, h_x.data(), sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z.data(), sizeof(double)*N, cudaMemcpyHostToDevice);

    void *d_p = myMalloc(sizeof(GpuParser<3>));
    GpuParser<3> *gpu_parser = new (d_p) GpuParser<3>(cpu_parser);
    cudaMemPrefetchAsync(gpu_parser, sizeof(GpuParser<3>), 0);
    cudaDeviceSynchronize();

    launch<<<(N+127)/128,128>>>([=] AMREX_GPU_DEVICE ()
    {
        int i = blockDim.x*blockIdx.x+threadIdx.x;
        d_r[i] = (*gpu_parser)(d_x[i], d_y[i], d_z[i]) - 5.0;
    });

    cudaMemcpy(h_r.data(), d_r, sizeof(double)*N, cudaMemcpyDeviceToHost);

    bool error = false;
    for (int i = 0; i < N; ++i) {
        error = error or (std::abs(h_r[i]) > 1.e-14);
    }
    if (error) {
        std::cout << "Wrong result" << std::endl;
    } else {
        std::cout << "Success" << std::endl;
    }

    gpu_parser->~GpuParser<3>();
    myFree(d_p);

    myFree(d_x);
    myFree(d_y);
    myFree(d_z);
    myFree(d_r);
}
