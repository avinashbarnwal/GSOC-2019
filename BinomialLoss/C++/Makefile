OS := $(shell uname)

ifeq ($(OS), Darwin)
	OUTPUT_LIB = distlib.dylib
else
	OUTPUT_LIB = distlib.so
endif

all: $(OUTPUT_LIB)

$(OUTPUT_LIB): neg_grad.cpp hessian.cpp loss.cpp
	$(CXX) -O3 -shared -o $@ $^
