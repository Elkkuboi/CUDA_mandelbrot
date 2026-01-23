# CUDA_mandelbrot

A straightforward Mandelbrot set renderer written in C++ and CUDA.

The calculation for 1M pixels is effectively instant. The implementation is "embarrassingly parallel"—every pixel is computed in its own independent thread with zero synchronization overhead.

### Key Points
* **Native C++ & CUDA:** No unnecessary wrappers.
* **pgm output:** Generates `.pgm`

### Usage
Requires Linux and the CUDA Toolkit (nvcc).

```bash
make
./mandelbrot

![Mandelbrot render](result.png)