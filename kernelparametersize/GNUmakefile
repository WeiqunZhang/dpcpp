DEBUG = FALSE

USE_DPCPP = TRUE
DPCPP_AOT = TRUE
DPCPP_SPLIT_KERNEL = FALSE

USE_MPI  = FALSE
USE_OMP  = FALSE

USE_HYPRE = FALSE
USE_PETSC = FALSE

COMP = gnu

DIM = 3

AMREX_HOME ?= ../../amrex

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package

Pdirs := Base Boundary Amr AmrCore

Ppack	+= $(foreach dir, $(Pdirs), $(AMREX_HOME)/Src/$(dir)/Make.package)

include $(Ppack)

include $(AMREX_HOME)/Tools/GNUMake/Make.rules

ifeq ($(DPCPP_SPLIT_KERNEL),TRUE)
  CXXFLAGS += -fsycl-device-code-split=per_kernel
endif


