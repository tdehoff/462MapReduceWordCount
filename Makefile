CC = g++
CFLAGS = -fopenmp -Wall -O3
LDFLAGS = -lm

all: seq omp

seq: sequential.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

omp: omp.cpp
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f seq omp *.txt
