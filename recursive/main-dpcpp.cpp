#include "GpuParser.H"
#include <CL/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <random>
#include <vector>
#include <cmath>

namespace sycl = cl::sycl;

namespace {
    std::unique_ptr<sycl::context> context;
    std::unique_ptr<sycl::device>  device;
    std::unique_ptr<sycl::queue>   queue;
}

void* myMalloc (std::size_t s)
{
    return sycl::malloc_shared(s, *device, *context);
}

void myFree (void* p)
{
    sycl::free(p, *context);
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

    sycl::gpu_selector my_selector;
    device.reset(new sycl::device(my_selector));
    context.reset(new sycl::context(*device));
    queue.reset(new sycl::queue(*context, my_selector,
                                sycl::property_list{sycl::property::queue::in_order{}}));

    double *d_x = (double*)myMalloc(sizeof(double)*N);
    double *d_y = (double*)myMalloc(sizeof(double)*N);
    double *d_z = (double*)myMalloc(sizeof(double)*N);
    double *d_r = (double*)myMalloc(sizeof(double)*N);

    queue->submit([&] (sycl::handler& h)
    {
        h.memcpy(d_x, h_x.data(), sizeof(double)*N);
    });
    queue->submit([&] (sycl::handler& h)
    {
        h.memcpy(d_y, h_y.data(), sizeof(double)*N);
    });
    queue->submit([&] (sycl::handler& h)
    {
        h.memcpy(d_z, h_z.data(), sizeof(double)*N);
    });

    void *d_p = myMalloc(sizeof(GpuParser<3>));
    GpuParser<3> *gpu_parser = new (d_p) GpuParser<3>(cpu_parser);
    queue->submit([&] (sycl::handler& h)
    {
        h.prefetch(gpu_parser, sizeof(GpuParser<3>));
    });

    queue->submit([&] (sycl::handler& h)
    {
        h.parallel_for(sycl::range<1>(N),
        [=] (sycl::item<1> item)
        {
            int i = item.get_linear_id();
//            d_r[i] = std::sqrt((d_x[i]*d_x[i]+d_y[i]*d_y[i]+d_z[i]*d_z[i])/(3.*3.))*5. - 5.0;
            d_r[i] = (*gpu_parser)(d_x[i], d_y[i], d_z[i]) - 5.0;
        });
    });

    queue->submit([&] (sycl::handler& h)
    {
        h.memcpy(h_r.data(), d_r, sizeof(double)*N);
    });

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

    queue.reset();
    context.reset();
    device.reset();
}
