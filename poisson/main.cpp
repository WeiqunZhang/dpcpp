
#include <AMReX.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        int const n_cell = 64;
        BoxArray ba(Box(IntVect{0}, IntVect{n_cell-1}));
        DistributionMapping dmap(ba);
        MultiFab solution(ba, dmap, 1, 1);
        solution.setVal(0.0);
    }

    amrex::Finalize();
}
