default: dpcpp

dpcpp: dpcpp-test.ex

cuda: cuda-test.ex

dpcpp-test.ex: dpcpp-test.cpp
	dpcpp -O3 -o $@ $<

cuda-test.ex: cuda-test.cpp
	nvcc -ccbin=g++ -Xcompiler='-O3 --std=c++14' --std=c++14 -Wno-deprecated-gpu-targets -m64 -arch=compute_60 -code=sm_60 -maxrregcount=255 --expt-relaxed-constexpr --expt-extended-lambda -x cu -o $@ $<

clean:
	${RM} *.ex *.o
