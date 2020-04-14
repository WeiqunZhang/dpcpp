DEBUG = FALSE

CRASH = TRUE

USE_DPCPP = TRUE

USE_MPI  = FALSE
USE_OMP  = FALSE

USE_HYPRE = FALSE
USE_PETSC = FALSE

COMP = gnu

DIM = 3

AMREX_HOME ?= ../../amrex

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

Pdirs := Base Boundary
ifeq ($(CRASH),TRUE)
  Pdirs += LinearSolvers/MLMG
  USERSuffix = .crash
endif

Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

