COMPILER = nvcc
CFLAGS = --compiler-options -Wall -I /usr/local/cuda-7.5/samples/common/inc -std=c++11
OBJECTS = csvloader.o
EXES = baaracuda

all: ${EXES}

csvloader.o: csvloader.cpp
	${COMPILER} ${CFLAGS} -c csvloader.cpp

baaracuda: main.cu csvloader.o
	${COMPILER} ${CFLAGS} main.cu ${OBJECTS} -o baaracuda

clean:
	rm -f *~ ${EXES} ${OBJECTS}
