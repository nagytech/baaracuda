COMPILER = nvcc
CFLAGS = -I /usr/local/cuda-7.5/samples/common/inc -std=c++11
EXES = baaracuda 

all: ${EXES}

baaracuda:  main.cu
	${COMPILER} ${CFLAGS} main.cu -o baaracuda

%.o: %.c %.h  makefile
	${COMPILER} ${CFLAGS} $< -c 

clean:
	rm -f *.o *~ ${EXES} ${CFILES}
