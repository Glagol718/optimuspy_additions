CFLAGS=-O3 -march=native
CL_CFLAGS=/O2 /arch:AVX2

all: gpp

gpp:
	g++ $(CFLAGS) -c catch_amalgamated.cpp -o catch_amalgamated_g++.o

clang:
	clang++-15 $(CFLAGS) -c catch_amalgamated.cpp -o catch_amalgamated_clang++-15.o

msvc:
	cl /EHsc $(CL_CFLAGS) /c catch_amalgamated.cpp /Fo:catch_amalgamated_cl.o
