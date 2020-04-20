DPCPP_DIR ?= /opt/intel/inteloneapi/compiler/latest/linux

CXXFLAGS = -fsycl -fsycl-targets=spir64_gen-unknown-unknown-sycldevice -Xsycl-target-backend '-device skl'
libraries = $(DPCPP_DIR)/lib/libsycl-cmath.o $(DPCPP_DIR)/lib/libsycl-cmath-fp64.o

default: a.out

a.out: main.cpp
	dpcpp $(CXXFLAGS) -o $@ $< $(libraries)

clean:
	${RM} a.out *.o
