
default: a.out

a.out: main.o random.o
	dpcpp -o $@ $^

main.o: main.cpp random.H
	dpcpp -c -o $@ $<

random.o: random.cpp
	dpcpp -c -o $@ $<

clean:
	${RM} a.out *.o

