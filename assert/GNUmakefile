DPCPP_DIR ?= /opt/intel/inteloneapi/compiler/latest/linux

default: a.out

a.out: main.cpp
	dpcpp -o $@ $< $(DPCPP_DIR)/lib/libsycl-glibc.o

clean:
	${RM} a.out *.o
