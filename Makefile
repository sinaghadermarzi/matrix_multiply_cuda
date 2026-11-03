NVCC      ?= nvcc
NVCCFLAGS ?= -O2

# The sources use the MSVC-only scanf_s; alias it to scanf on other hosts.
ifneq ($(OS),Windows_NT)
NVCCFLAGS += -Dscanf_s=scanf
endif

BINARIES = 1D_phase1 2D_phase1 1D_phase2 2D_phase2

all: $(BINARIES)

1D_phase1: 1D_threads(1st_phase).cu
	$(NVCC) $(NVCCFLAGS) -o $@ "$<"

2D_phase1: 2D_threads(1st_phase).cu
	$(NVCC) $(NVCCFLAGS) -o $@ "$<"

1D_phase2: 1D_threads(2nd_phase).cu
	$(NVCC) $(NVCCFLAGS) -o $@ "$<"

2D_phase2: 2D_threads(2nd_phase).cu
	$(NVCC) $(NVCCFLAGS) -o $@ "$<"

clean:
	rm -f $(BINARIES)

.PHONY: all clean
