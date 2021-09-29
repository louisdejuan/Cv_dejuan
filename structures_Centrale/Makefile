CXX=g++
CXXFLAGS=-std=c++2a -Wall -g -D_GLIBBCXX_DEBUG -fconcepts
LDFLAGS=
EXEC=exec
SRC=$(wildcard *.cpp)
HDR=$(wildcard *.hpp)
ARGS=

all: $(EXEC)

$(EXEC): $(patsubst %.cpp,%.o,$(SRC)) $(HDR)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $^

%.o: %.cpp $(HDR)
	$(CXX) -o $@ -c $(CXXFLAGS) $<

clean: $(HDR) $(SRC)
	rm -Rf *.o $(EXEC)

run: $(EXEC)
	./$(EXEC) $(ARGS)

debug: $(EXEC)
	gdb -tui --args ./$(EXEC) $(ARGS)
