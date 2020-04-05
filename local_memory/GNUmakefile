
default: a.out

a.out: main.o
	dpcpp -o $@ $^

main.o: main.cpp
	dpcpp -c -o $@ $<

clean:
	${RM} a.out *.o

