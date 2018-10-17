NVCC_FLAGS = -std=c++11 -lcublas
NVCC = nvcc $(CHARM_INC) $(NVCC_FLAGS)

# include optimization

all: outer

outer: outer.cu
	$(NVCC) -O3 -o $@ $<

clean:
	rm -f outer

test: all
	./outer
