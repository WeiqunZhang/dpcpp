
#include <AMReX.H>
#include <vector>
#include <random>

#define CRASH 1  // It runs if this is changeg to 0.

namespace {
#if (CRASH == 1)
    std::mt19937 generator;
#endif
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

#if (CRASH != 1)
    std::mt19937 generator;
#endif

    generator.seed(567);

    long N = 256;
    std::vector<double> hv(N);

    for (long i = 0; i < N; ++i) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        hv[i] = distribution(generator);
    }

    amrex::Finalize();
}

