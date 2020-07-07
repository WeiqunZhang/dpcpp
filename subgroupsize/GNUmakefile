
default: a.out

a.out: main.cpp
	dpcpp -device-math-lib=fp32,fp64 -o $@ $<

clean:
	${RM} a.out *.o
