CXX      := g++
CXXFLAGS := -O3 -std=c++17 -pthread -march=native
TESTFLAGS := -O1 -g -std=c++17 -pthread -march=native

.PHONY: all test test-unit test-integration clean

# ---------------------------------------------------------------------------
# Default target: build the main binary
# ---------------------------------------------------------------------------
all: hypersketch

hypersketch: hypersketch.cpp
	$(CXX) $(CXXFLAGS) hypersketch.cpp -o hypersketch

# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------

## Download Catch2 v2 single-header on first use
tests/catch.hpp:
	@echo "Downloading Catch2 v2 single header..."
	curl -sL https://github.com/catchorg/Catch2/releases/download/v2.13.10/catch.hpp \
	     -o tests/catch.hpp

## Build and run C++ unit tests
tests/test_unit: tests/test_unit.cpp hypersketch.cpp tests/catch.hpp
	$(CXX) $(TESTFLAGS) -DHYPERSKETCH_TEST_BUILD tests/test_unit.cpp -o tests/test_unit

test-unit: tests/test_unit
	./tests/test_unit

## Build hypersketch and run Python integration tests
test-integration: hypersketch
	python3 -m pytest tests/test_integration.py -v

## Run all tests
test: test-unit test-integration

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
clean:
	rm -f hypersketch tests/test_unit tests/fixtures/*.hms
