default: dpcpp

dpcpp: dpcpp-test.ex

cuda: cuda-test.ex

dpcpp-test.ex: dpcpp-test.cpp
	dpcpp -O3 -o $@ $<

cuda-test.ex: cuda-test.cpp
	nvcc -O3 -o $@ $<

clean:
	${RM} *.ex *.o
