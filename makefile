COMPILER = nvcc
CFLAGS = --compiler-options -Wall#-I /usr/local/cuda-7.5/samples/common/inc -std=c++11
OBJECTS = csvloader.o kernels.o runner.o
EXES = baaracuda

all: ${EXES}

profile: CFLAGS+=-DPROFILE
profile: all

csvloader.o: csvloader.cpp
	${COMPILER} ${CFLAGS} -c csvloader.cpp

runner.o: runner.cu kernels.o
	${COMPILER} ${CFLAGS} -c runner.cu kernels.o

kernels.o: kernels.cu
	${COMPILER} ${CFLAGS} -c kernels.cu

baaracuda: main.cu ${OBJECTS}
	${COMPILER} ${CFLAGS} main.cu ${OBJECTS} -o baaracuda

clean:
	rm -f *~ ${EXES} ${OBJECTS}
