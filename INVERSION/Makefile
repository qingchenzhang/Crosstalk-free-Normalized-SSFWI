#! bin/sh

MPICH2_HOME=/home/Software/Intel/impi/5.0.1.035/intel64
CUDA_HOME=/usr/local/cuda-9.0

NVCC = $(CUDA_HOME)/bin/nvcc
MPICC = $(MPICH2_HOME)/bin/mpicc

INC = -I$(CUDA_HOME)/include -I$(MPICH2_HOME)/include -I$(MADAG_HOME)/include -I$(FFTW_HOME)/include
LIB = -L$(CUDA_HOME)/lib64  -L$(MPICH2_HOME)/lib -L$(MADAG_HOME)/lib -L$(FFTW_HOME)/lib64

LDFLAGS = -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX -lmpi -lrt -lpthread -Xcompiler -fopenmp -lm -lstdc++ -lm -lgomp -lfftw3 -lfftw3f -lfftw3f_threads -lfftw3_threads -lcudart -lcurand -lcufft #-lblas -llapack -lblas

CPPFILE = acoustic_HLMPIFWIC.cpp
CUFILE  = acoustic_HLMPIFWIMulti.cu
OBJECTS = acoustic_HLMPIFWIC.o acoustic_HLMPIFWIMulti.o
EXECNAME = fwiacoustic

all:
	mpicc -w -c -g -x c -std=gnu99 -Wall -pedantic -fopenmp -lfftw3 -DNO_BLAS  -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX $(INC) $(CPPFILE)
	nvcc -w -c $(INC) $(CUFILE) $(LIB) $(LDFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
	nvcc -w -o $(EXECNAME) $(OBJECTS) $(INC) $(LIB) $(LDFLAGS) -DMPICH_IGNORE_CXX_SEEK -DMPICH_SKIP_MPICXX
	rm -f *.o 

#CFILES = acoustic_HLMPIFWIC.cpp
#CUFILES = acoustic_HLMPIFWIMulti.cu
#EXECNAME = fwiacoustic_reg

#all:
#	$(NVCC) -w -v -o $(EXECNAME) $(CFILES) $(CUFILES) $(INC) $(LIB) $(LDFLAGS)
#	rm -f *.o 


