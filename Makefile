# Kääntäjä
NVCC = nvcc

# Liput: Optimoidaan RTX 5090:lle (native) ja käytetään C++11 standardia
FLAGS = -arch=native -std=c++11 -O3

# Kohdetiedosto
TARGET = mandelbrot

# Lähdekoodi
SRC = src/mandelbrot.cu

all:
	$(NVCC) $(FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
