CXX      := g++
CXXFLAGS := -O3 -std=c++17 -pthread -march=native
TESTFLAGS := -O1 -g -std=c++17 -pthread -march=native

.PHONY: all test test-unit test-integration clean

# ---------------------------------------------------------------------------
# Default target: build the main binary
# ---------------------------------------------------------------------------
all: hypermash

hypermash: hypermash.cpp
	$(CXX) $(CXXFLAGS) hypermash.cpp -o hypermash

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------

## Download Catch2 v2 single-header on first use
tests/catch.hpp:
	@echo "Downloading Catch2 v2 single header..."
	curl -sL https://github.com/catchorg/Catch2/releases/download/v2.13.10/catch.hpp \
	     -o tests/catch.hpp

## Build and run C++ unit tests
tests/test_unit: tests/test_unit.cpp hypermash.cpp tests/catch.hpp
	$(CXX) $(TESTFLAGS) -DHYPERMASH_TEST_BUILD tests/test_unit.cpp -o tests/test_unit

test-unit: tests/test_unit
	./tests/test_unit

## Build hypermash and run Python integration tests
test-integration: hypermash
	python3 -m pytest tests/test_integration.py -v

## Run all tests
test: test-unit test-integration

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	rm -f hypermash tests/test_unit tests/fixtures/*.hms
