SETTING=-O3 -std=c++17 -larmadillo -lmlpack -lboost_serialization -fopenmp -fpic -march=native -mavx512f

SOURCE=$(wildcard ./src/*.cpp)

suco:$(SRCS)
	rm -rf taco
	g++ $(SOURCE) -o taco $(SETTING)

clean:
	rm -rf taco
