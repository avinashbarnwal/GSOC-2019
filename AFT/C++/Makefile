OS := $(shell uname)
ifeq ($(OS), Darwin)
	OUTPUT_LIB = distlib.dylib
else
	OUTPUT_LIB = distlib.so
endif

all: $(OUTPUT_LIB)

$(OUTPUT_LIB): dlogis.cpp dnorm.cpp plogis.cpp pnorm.cpp grad_norm.cpp grad_logis.cpp hess_norm.cpp hess_logis.cpp loss_uncensored.cpp loss_left.cpp loss_right.cpp loss_interval.cpp neg_grad_uncensored.cpp neg_grad_left.cpp neg_grad_right.cpp neg_grad_interval.cpp hessian_uncensored.cpp hessian_left.cpp hessian_right.cpp hessian_interval.cpp
	$(CXX) -O3 -shared -o $@ $^
