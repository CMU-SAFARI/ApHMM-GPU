CXX=g++
RM=rm -rf

#include directories seperated by a space
INC=./utils/include/
INC_PARAMS=$(foreach d, $(INC), -I$d)
CUDAINC=/usr/local/cuda/include/ /usr/local/cuda/samples/common/inc/
CUDAINC_PARAMS=$(foreach d, $(CUDAINC), -I$d)

#lib directories seperated by a space
LIB=./utils/lib/
LIB_PARAMS=$(foreach d, $(LIB), -L$d)
CUDALIB=/usr/local/cuda/lib64/ /usr/local/cuda/samples/common/lib/
CUDALIB_PARAMS=$(foreach d, $(CUDALIB), -L$d)

#paramaters as suggested by seqan
CXXLAGS=-std=c++14 -O3 -DNDEBUG -DSEQAN_HAS_ZLIB=1 -DSEQAN_HAS_BZIP2=1 -DSEQAN_DISABLE_VERSION_CHECK=YES -W -Wall -pedantic
#Parameters for debug purposes. Comment the above line and out the below one to compile in debug mode
#CXXLAGS=$(INC_PARAMS) -std=c++14 -g -O0 -DSEQAN_ENABLE_DEBUG=1 -DSEQAN_HAS_ZLIB=1 -DSEQAN_HAS_BZIP2=1 -DSEQAN_DISABLE_VERSION_CHECK=YES -W -Wall -pedantic

LDFLAGS=
LDLIBS=$(LIB_PARAMS) -lz -lbz2
CUDALDLIBS=$(CUDALIB_PARAMS) -lcuda -lcudart

ifeq ($(OS),Windows_NT)
    CXXLAGS += -D WIN32
    ifeq ($(PROCESSOR_ARCHITEW6432),AMD64)
        CXXLAGS+=/W2 /wd4996 -D_CRT_SECURE_NO_WARNINGS
    else
        ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
            CXXLAGS+=/W2 /wd4996 -D_CRT_SECURE_NO_WARNINGS
        endif
        ifeq ($(PROCESSOR_ARCHITECTURE),x86)
            CXXLAGS+=/W2 /wd4996 -D_CRT_SECURE_NO_WARNINGS
        endif
    endif
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        CXXLAGS += -fopenmp
        LDLIBS += -lpthread -lrt
	CUDALDLIBS += -lgomp
    endif
    ifeq ($(UNAME_S),Darwin)
        CXXLAGS +=
    endif
endif


# CODE ADDED FOR CUDA
#EXEC ?= @echo "[@]"
CUDA_PATH?=/usr/local/cuda
NVCC=$(CUDA_PATH)/bin/nvcc -ccbin $(CXX)
NVCCFLAGS=-m64 # TODO: evt need more things here
# Gencode arguments
#SMS ?= 70 (titan v) 80 (a100)
SM=70
#GENCODE_FLAGS=$(foreach g, $(SM), $(eval GENCODE_FLAGS+=-gencode arch=compute_$(g), code=sm_$(g)))
GENCODE_FLAGS+=-gencode arch=compute_$(SM),code=compute_$(SM)

ALL_CCFLAGS:=
ALL_CCFLAGS+=$(NVCCFLAGS)
# ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS+=$(addprefix -Xcompiler ,$(CXXLAGS))
### END CODE FOR CUDA


#can add many src files if it is necessary in future (by putting spaces)
SRC= ./src/HMMDecoder.cpp ./src/HMMTrainer.cu ./src/HMMGraph.cpp ./src/Polisher.cpp ./src/main.cpp
OBJS1=$(subst .cpp,.o,$(SRC))
OBJS=$(subst .cu,.o,$(OBJS1))

default: all

all: $(OBJS)
	mkdir -p ./bin/
	$(CXX) $(LDFLAGS) -Lbin/static -o ./bin/aphmm-gpu $(OBJS) ./src/HMMTrainer.co $(LDLIBS) $(CUDALDLIBS)

./src/HMMGraph.o: ./src/HMMGraph.cpp ./src/HMMGraph.h ./src/HMMCommons.h ./utils/lib/libz.so ./utils/lib/libbz2.so ./utils/include/seqan/basic.h
	$(CXX) $(INC_PARAMS) $(CXXLAGS) -c ./src/HMMGraph.cpp -o ./src/HMMGraph.o
#	$(CXX) $(INC_PARAMS) $(CXXLAGS) -fPIC ./src/HMMGraph.cpp ./src/HMMCommons.h -shared -o ./src/libHMMGraph.so -Wl,--whole-archive $(LDLIBS) -Wl,--no-whole-archive

./src/HMMDecoder.o: ./src/HMMGraph.o ./src/HMMDecoder.cpp ./src/HMMDecoder.h
	$(CXX) $(INC_PARAMS) $(CXXLAGS) -c -o ./src/HMMDecoder.o ./src/HMMDecoder.cpp

./src/HMMTrainer.o: ./src/HMMGraph.o ./src/HMMTrainer.cu ./src/HMMTrainer.h
	$(CXX) $(INC_PARAMS) $(CUDAINC_PARAMS) $(CXXLAGS) -c -o ./src/HMMTrainer.o ./src/HMMTrainer.cpp
	echo "executing NVCC compilation"
	$(EXEC) $(NVCC) $(CUDAINC_PARAMS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o ./src/HMMTrainer.co -c ./src/HMMTrainer.cu

./src/Polisher.o: ./src/HMMDecoder.o ./src/HMMTrainer.o ./src/HMMGraph.o ./src/Polisher.cpp ./src/Polisher.h
	$(CXX) $(INC_PARAMS) $(CXXLAGS) -c -o ./src/Polisher.o ./src/Polisher.cpp

./src/main.o: ./src/Polisher.o ./src/main.cpp ./src/CommandLineParser.h
	$(CXX) $(INC_PARAMS) $(CXXLAGS) -c -o ./src/main.o ./src/main.cpp

./utils/lib/libz.so: ./utils/zlib-1.2.11.tar.gz
	mkdir -p ./utils/include/
	mkdir -p ./utils/lib/
	tar -xf ./utils/zlib-1.2.11.tar.gz -C ./utils/
	cd ./utils/zlib-1.2.11/; prefix=./ ./configure --sharedlibdir=../lib/ --64; make install prefix=../
	rm -rf ./utils/zlib-1.2.11/

./utils/lib/libbz2.so: ./utils/bzip2-1.0.6.tar.gz
	mkdir -p ./utils/include/
	mkdir -p ./utils/lib/
	tar -xf ./utils/bzip2-1.0.6.tar.gz -C ./utils/
	cd ./utils/bzip2-1.0.6/; make -f Makefile-libbz2_so; mv libbz2.so.1.0.6 ../lib/; mv libbz2.so.1.0 ../lib/libbz2.so; make -f Makefile-libbz2_so clean; make install PREFIX=../
	rm -rf ./utils/bzip2-1.0.6/

./utils/include/seqan/basic.h: ./utils/seqan.tar.gz
	rm -rf ./utils/include/seqan/
	mkdir -p ./utils/include/
	mkdir -p ./utils/lib/
	tar -xzf ./utils/seqan.tar.gz -C ./utils/
	mv ./utils/seqan ./utils/include/
# 	rm -rf ./utils/seqan-library-2.4.0

clean:
	$(RM) $(OBJS) ./bin/ ./utils/include/ ./utils/lib/ ./utils/share/ ./utils/man/ ./utils/bin/


