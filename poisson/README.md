This test reproduces a bug in Beta5.

Compiling with `make` and running with `./main3d.dpcpp.crash.ex` will
result in segfault.

Compiling with `make CRASH=FALSE` and running with `./main3d.dpcpp.ex`
will not result in segfault.

The difference between the two is that the former includes amrex
source code under `amrex/Src/LinearSolvers/MLMG/`, whereas the latter
does not.  Note that codes in `amrex/Src/LinearSolvers/MLMG/` will not
be used at all in this test.  But linking them into the executable
will make the run crash.
