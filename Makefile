# 5G NR PUSCH scrambler - Linux user space.
#   make                 build with Intel oneAPI icpx (default)
#   make CXX=g++         build with GCC instead
#   make test            self-test + known-answer vectors
#   make bench           throughput report
#   make vectors         regenerate test vectors (needs python3)

CXX      ?= icpx
OPT      ?= -O3
ARCH     ?= -mavx2
WARN     ?= -Wall -Wextra -Werror
CXXFLAGS ?= -std=c++17 $(OPT) $(ARCH) $(WARN)
INCLUDES := -Iinclude

TARGET := pusch_scrambler
SRCS   := src/main.cpp src/io.cpp src/prbs.cpp src/scrambler.cpp
OBJS   := $(SRCS:src/%.cpp=build/%.o)
DEPS   := $(OBJS:.o=.d)

.PHONY: all test bench vectors clean
all: $(TARGET)

build:
	mkdir -p build

build/%.o: src/%.cpp | build
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -c -o $@ $<

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

test: $(TARGET)
	./$(TARGET) selftest
	./test/run_tests.sh

bench: $(TARGET)
	./$(TARGET) bench

vectors:
	python3 tools/gen_vectors.py test/vectors

clean:
	rm -rf build $(TARGET)

-include $(DEPS)
