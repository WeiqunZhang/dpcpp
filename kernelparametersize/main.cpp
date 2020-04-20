
#include <AMReX.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

template <typename T>
void mlndlap_bc_doit (Box const& vbx, Array4<T> const& a, Box const& domain,
                      GpuArray<bool,AMREX_SPACEDIM> const& bflo,
                      GpuArray<bool,AMREX_SPACEDIM> const& bfhi) noexcept
{
    Box gdomain = domain;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (not bflo[idim]) gdomain.growLo(idim,1);
        if (not bfhi[idim]) gdomain.growHi(idim,1);
    }

    if (gdomain.strictly_contains(vbx)) return;

    int offset = domain.cellCentered() ? 0 : 1;

    const auto dlo = amrex::lbound(domain);
    const auto dhi = amrex::ubound(domain);

    Box const& sbox = amrex::grow(vbx,1);
    AMREX_HOST_DEVICE_FOR_3D(sbox, i, j, k,
    {
        if (not gdomain.contains(IntVect(i,j,k))) {
            // xlo & ylo & zlo
            if (i == dlo.x-1 and j == dlo.y-1 and k == dlo.z-1 and (bflo[0] or bflo[1] or bflo[2]))
            {
                if (bflo[0] and bflo[1] and bflo[2])
                {
                    a(i,j,k) = a(i+1+offset, j+1+offset, k+1+offset);
                }
                else if (bflo[0] and bflo[1])
                {
                    a(i,j,k) = a(i+1+offset, j+1+offset, k);
                }
                else if (bflo[0] and bflo[2])
                {
                    a(i,j,k) = a(i+1+offset, j, k+1+offset);
                }
                else if (bflo[1] and bflo[2])
                {
                    a(i,j,k) = a(i, j+1+offset, k+1+offset);
                }
                else if (bflo[0])
                {
                    a(i,j,k) = a(i+1+offset, j, k);
                }
                else if (bflo[1])
                {
                    a(i,j,k) = a(i, j+1+offset, k);
                }
                else if (bflo[2])
                {
                    a(i,j,k) = a(i, j, k+1+offset);
                }
            }
            // xhi & ylo & zlo
            else if (i == dhi.x+1 and j == dlo.y-1 and k == dlo.z-1 and (bfhi[0] or bflo[1] or bflo[2]))
            {
                if (bfhi[0] and bflo[1] and bflo[2])
                {
                    a(i,j,k) = a(i-1-offset, j+1+offset, k+1+offset);
                }
                else if (bfhi[0] and bflo[1])
                {
                    a(i,j,k) = a(i-1-offset, j+1+offset, k);
                }
                else if (bfhi[0] and bflo[2])
                {
                    a(i,j,k) = a(i-1-offset, j, k+1+offset);
                }
                else if (bflo[1] and bflo[2])
                {
                    a(i,j,k) = a(i, j+1+offset, k+1+offset);
                }
                else if (bfhi[0])
                {
                    a(i,j,k) = a(i-1-offset, j, k);
                }
                else if (bflo[1])
                {
                    a(i,j,k) = a(i, j+1+offset, k);
                }
                else if (bflo[2])
                {
                    a(i,j,k) = a(i, j, k+1+offset);
                }
            }
            // xlo & yhi & zlo
            else if (i == dlo.x-1 and j == dhi.y+1 and k == dlo.z-1 and (bflo[0] or bfhi[1] or bflo[2]))
            {
                if (bflo[0] and bfhi[1] and bflo[2])
                {
                    a(i,j,k) = a(i+1+offset, j-1-offset, k+1+offset);
                }
                else if (bflo[0] and bfhi[1])
                {
                    a(i,j,k) = a(i+1+offset, j-1-offset, k);
                }
                else if (bflo[0] and bflo[2])
                {
                    a(i,j,k) = a(i+1+offset, j, k+1+offset);
                }
                else if (bfhi[1] and bflo[2])
                {
                    a(i,j,k) = a(i, j-1-offset, k+1+offset);
                }
                else if (bflo[0])
                {
                    a(i,j,k) = a(i+1+offset, j, k);
                }
                else if (bfhi[1])
                {
                    a(i,j,k) = a(i, j-1-offset, k);
                }
                else if (bflo[2])
                {
                    a(i,j,k) = a(i, j, k+1+offset);
                }
            }
            // xhi & yhi & zlo
            else if (i == dhi.x+1 and j == dhi.y+1 and k == dlo.z-1 and (bfhi[0] or bfhi[1] or bflo[2]))
            {
                if (bfhi[0] and bfhi[1] and bflo[2])
                {
                    a(i,j,k) = a(i-1-offset, j-1-offset, k+1+offset);
                }
                else if (bfhi[0] and bfhi[1])
                {
                    a(i,j,k) = a(i-1-offset, j-1-offset, k);
                }
                else if (bfhi[0] and bflo[2])
                {
                    a(i,j,k) = a(i-1-offset, j, k+1+offset);
                }
                else if (bfhi[1] and bflo[2])
                {
                    a(i,j,k) = a(i, j-1-offset, k+1+offset);
                }
                else if (bfhi[0])
                {
                    a(i,j,k) = a(i-1-offset, j, k);
                }
                else if (bfhi[1])
                {
                    a(i,j,k) = a(i, j-1-offset, k);
                }
                else if (bflo[2])
                {
                    a(i,j,k) = a(i, j, k+1+offset);
                }
            }
            // xlo & ylo & zhi
            else if (i == dlo.x-1 and j == dlo.y-1 and k == dhi.z+1 and (bflo[0] or bflo[1] or bfhi[2]))
            {
                if (bflo[0] and bflo[1] and bfhi[2])
                {
                    a(i,j,k) = a(i+1+offset, j+1+offset, k-1-offset);
                }
                else if (bflo[0] and bflo[1])
                {
                    a(i,j,k) = a(i+1+offset, j+1+offset, k);
                }
                else if (bflo[0] and bfhi[2])
                {
                    a(i,j,k) = a(i+1+offset, j, k-1-offset);
                }
                else if (bflo[1] and bfhi[2])
                {
                    a(i,j,k) = a(i, j+1+offset, k-1-offset);
                }
                else if (bflo[0])
                {
                    a(i,j,k) = a(i+1+offset, j, k);
                }
                else if (bflo[1])
                {
                    a(i,j,k) = a(i, j+1+offset, k);
                }
                else if (bfhi[2])
                {
                    a(i,j,k) = a(i, j, k-1-offset);
                }
            }
            // xhi & ylo & zhi
            else if (i == dhi.x+1 and j == dlo.y-1 and k == dhi.z+1 and (bfhi[0] or bflo[1] or bfhi[2]))
            {
                if (bfhi[0] and bflo[1] and bfhi[2])
                {
                    a(i,j,k) = a(i-1-offset, j+1+offset, k-1-offset);
                }
                else if (bfhi[0] and bflo[1])
                {
                    a(i,j,k) = a(i-1-offset, j+1+offset, k);
                }
                else if (bfhi[0] and bfhi[2])
                {
                    a(i,j,k) = a(i-1-offset, j, k-1-offset);
                }
                else if (bflo[1] and bfhi[2])
                {
                    a(i,j,k) = a(i, j+1+offset, k-1-offset);
                }
                else if (bfhi[0])
                {
                    a(i,j,k) = a(i-1-offset, j, k);
                }
                else if (bflo[1])
                {
                    a(i,j,k) = a(i, j+1+offset, k);
                }
                else if (bfhi[2])
                {
                    a(i,j,k) = a(i, j, k-1-offset);
                }
            }
            // xlo & yhi & zhi
            else if (i == dlo.x-1 and j == dhi.y+1 and k == dhi.z+1 and (bflo[0] or bfhi[1] or bfhi[2]))
            {
                if (bflo[0] and bfhi[1] and bfhi[2])
                {
                    a(i,j,k) = a(i+1+offset, j-1-offset, k-1-offset);
                }
                else if (bflo[0] and bfhi[1])
                {
                    a(i,j,k) = a(i+1+offset, j-1-offset, k);
                }
                else if (bflo[0] and bfhi[2])
                {
                    a(i,j,k) = a(i+1+offset, j, k-1-offset);
                }
                else if (bfhi[1] and bfhi[2])
                {
                    a(i,j,k) = a(i, j-1-offset, k-1-offset);
                }
                else if (bflo[0])
                {
                    a(i,j,k) = a(i+1+offset, j, k);
                }
                else if (bfhi[1])
                {
                    a(i,j,k) = a(i, j-1-offset, k);
                }
                else if (bfhi[2])
                {
                    a(i,j,k) = a(i, j, k-1-offset);
                }
            }
            // xhi & yhi & zhi
            else if (i == dhi.x+1 and j == dhi.y+1 and k == dhi.z+1 and (bfhi[0] or bfhi[1] or bfhi[2]))
            {
                if (bfhi[0] and bfhi[1] and bfhi[2])
                {
                    a(i,j,k) = a(i-1-offset, j-1-offset, k-1-offset);
                }
                else if (bfhi[0] and bfhi[1])
                {
                    a(i,j,k) = a(i-1-offset, j-1-offset, k);
                }
                else if (bfhi[0] and bfhi[2])
                {
                    a(i,j,k) = a(i-1-offset, j, k-1-offset);
                }
                else if (bfhi[1] and bfhi[2])
                {
                    a(i,j,k) = a(i, j-1-offset, k-1-offset);
                }
                else if (bfhi[0])
                {
                    a(i,j,k) = a(i-1-offset, j, k);
                }
                else if (bfhi[1])
                {
                    a(i,j,k) = a(i, j-1-offset, k);
                }
                else if (bfhi[2])
                {
                    a(i,j,k) = a(i, j, k-1-offset);
                }
            }
            // xlo & ylo
            else if (i == dlo.x-1 and j == dlo.y-1 and (bflo[0] or bflo[1]))
            {
                if (bflo[0] and bflo[1])
                {
                    a(i,j,k) = a(i+1+offset, j+1+offset, k);
                }
                else if (bflo[0])
                {
                    a(i,j,k) = a(i+1+offset, j, k);
                }
                else if (bflo[1])
                {
                    a(i,j,k) = a(i, j+1+offset, k);
                }
            }
            // xhi & ylo
            else if (i == dhi.x+1 and j == dlo.y-1 and (bfhi[0] or bflo[1]))
            {
                if (bfhi[0] and bflo[1])
                {
                    a(i,j,k) = a(i-1-offset, j+1+offset, k);
                }
                else if (bfhi[0])
                {
                    a(i,j,k) = a(i-1-offset, j, k);
                }
                else if (bflo[1])
                {
                    a(i,j,k) = a(i, j+1+offset, k);
                }
            }
            // xlo & yhi
            else if (i == dlo.x-1 and j == dhi.y+1 and (bflo[0] or bfhi[1]))
            {
                if (bflo[0] and bfhi[1])
                {
                    a(i,j,k) = a(i+1+offset, j-1-offset, k);
                }
                else if (bflo[0])
                {
                    a(i,j,k) = a(i+1+offset, j, k);
                }
                else if (bfhi[1])
                {
                    a(i,j,k) = a(i, j-1-offset, k);
                }
            }
            // xhi & yhi
            else if (i == dhi.x+1 and j == dhi.y+1 and (bfhi[0] or bfhi[1]))
            {
                if (bfhi[0] and bfhi[1])
                {
                    a(i,j,k) = a(i-1-offset, j-1-offset, k);
                }
                else if (bfhi[0])
                {
                    a(i,j,k) = a(i-1-offset, j, k);
                }
                else if (bfhi[1])
                {
                    a(i,j,k) = a(i, j-1-offset, k);
                }
            }
            // xlo & zlo
            else if (i == dlo.x-1 and k == dlo.z-1 and (bflo[0] or bflo[2]))
            {
                if (bflo[0] and bflo[2])
                {
                    a(i,j,k) = a(i+1+offset, j, k+1+offset);
                }
                else if (bflo[0])
                {
                    a(i,j,k) = a(i+1+offset, j, k);
                }
                else if (bflo[2])
                {
                    a(i,j,k) = a(i, j, k+1+offset);
                }
            }
            // xhi & zlo
            else if (i == dhi.x+1 and k == dlo.z-1 and (bfhi[0] or bflo[2]))
            {
                if (bfhi[0] and bflo[2])
                {
                    a(i,j,k) = a(i-1-offset, j, k+1+offset);
                }
                else if (bfhi[0])
                {
                    a(i,j,k) = a(i-1-offset, j, k);
                }
                else if (bflo[2])
                {
                    a(i,j,k) = a(i, j, k+1+offset);
                }
            }
            // xlo & zhi
            else if (i == dlo.x-1 and k == dhi.z+1 and (bflo[0] or bfhi[2]))
            {
                if (bflo[0] and bfhi[2])
                {
                    a(i,j,k) = a(i+1+offset, j, k-1-offset);
                }
                else if (bflo[0])
                {
                    a(i,j,k) = a(i+1+offset, j, k);
                }
                else if (bfhi[2])
                {
                    a(i,j,k) = a(i, j, k-1-offset);
                }
            }
            // xhi & zhi
            else if (i == dhi.x+1 and k == dhi.z+1 and (bfhi[0] or bfhi[2]))
            {
                if (bfhi[0] and bfhi[2])
                {
                    a(i,j,k) = a(i-1-offset, j, k-1-offset);
                }
                else if (bfhi[0])
                {
                    a(i,j,k) = a(i-1-offset, j, k);
                }
                else if (bfhi[2])
                {
                    a(i,j,k) = a(i, j, k-1-offset);
                }
            }
            // ylo & zlo
            else if (j == dlo.y-1 and k == dlo.z-1 and (bflo[1] or bflo[2]))
            {
                if (bflo[1] and bflo[2])
                {
                    a(i,j,k) = a(i, j+1+offset, k+1+offset);
                }
                else if (bflo[1])
                {
                    a(i,j,k) = a(i, j+1+offset, k);
                }
                else if (bflo[2])
                {
                    a(i,j,k) = a(i, j, k+1+offset);
                }
            }
            // yhi & zlo
            else if (j == dhi.y+1 and k == dlo.z-1 and (bfhi[1] or bflo[2]))
            {
                if (bfhi[1] and bflo[2])
                {
                    a(i,j,k) = a(i, j-1-offset, k+1+offset);
                }
                else if (bfhi[1])
                {
                    a(i,j,k) = a(i, j-1-offset, k);
                }
                else if (bflo[2])
                {
                    a(i,j,k) = a(i, j, k+1+offset);
                }
            }
            // ylo & zhi
            else if (j == dlo.y-1 and k == dhi.z+1 and (bflo[1] or bfhi[2]))
            {
                if (bflo[1] and bfhi[2])
                {
                    a(i,j,k) = a(i, j+1+offset, k-1-offset);
                }
                else if (bflo[1])
                {
                    a(i,j,k) = a(i, j+1+offset, k);
                }
                else if (bfhi[2])
                {
                    a(i,j,k) = a(i, j, k-1-offset);
                }
            }
            // yhi & zhi
            else if (j == dhi.y+1 and k == dhi.z+1 and (bfhi[1] or bfhi[2]))
            {
                if (bfhi[1] and bfhi[2])
                {
                    a(i,j,k) = a(i, j-1-offset, k-1-offset);
                }
                else if (bfhi[1])
                {
                    a(i,j,k) = a(i, j-1-offset, k);
                }
                else if (bfhi[2])
                {
                    a(i,j,k) = a(i, j, k-1-offset);
                }
            }
            else if (i == dlo.x-1 and bflo[0])
            {
                a(i,j,k) = a(i+1+offset, j, k);
            }
            else if (i == dhi.x+1 and bfhi[0])
            {
                a(i,j,k) = a(i-1-offset, j, k);
            }
            else if (j == dlo.y-1 and bflo[1])
            {
                a(i,j,k) = a(i, j+1+offset, k);
            }
            else if (j == dhi.y+1 and bfhi[1])
            {
                a(i,j,k) = a(i, j-1-offset, k);
            }
            else if (k == dlo.z-1 and bflo[2])
            {
                a(i,j,k) = a(i, j, k+1+offset);
            }
            else if (k == dhi.z+1 and bfhi[2])
            {
                a(i,j,k) = a(i, j, k-1-offset);
            }
        }
    });
}

template void mlndlap_bc_doit<Real> (Box const& vbx, Array4<Real> const& a, Box const& domain, GpuArray<bool,AMREX_SPACEDIM> const& bflo, GpuArray<bool,AMREX_SPACEDIM> const& bfhi) noexcept;

int main (int argc, char* argv[])
{
}
