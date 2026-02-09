# Nvidia Cuda Compiler
NVCC = /usr/local/cuda/bin/nvcc

# compiler flags are instructions
# -O3 is optimizer level, 3 is good for 5090
# -lsdl2 means link simple directmedia layer
# architecture native tells to use stuff made for my computer
CFLAGS = -O3 -lSDL2 -arch=native

# Where is code
SRC_DIR = src

# Objectfiles, cpp code is not straight translated to program but first an object
OBJ = main.o kernel.o
TARGET = mandelbrot

# Main rules, what to do : what needed, command
$(TARGET): $(OBJ)
	$(NVCC) $(OBJ) -o $(TARGET) $(CFLAGS)

# I want to make file mandelbrot, I need main and kernel


# I want to make main.o
main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/mandelbrot.h
	$(NVCC) -c $(SRC_DIR)/main.cpp $(CFLAGS)
# -c means compile only, doesn't try to write a program yet


# This creates kernel.o
kernel.o: $(SRC_DIR)/kernel.cu $(SRC_DIR)/mandelbrot.h
	$(NVCC) -c $(SRC_DIR)/kernel.cu $(CFLAGS)

# cleaning
clean:
	rm -f *.o $(TARGET)

# remove force, *.o temporary objects, deletes mandelbrot