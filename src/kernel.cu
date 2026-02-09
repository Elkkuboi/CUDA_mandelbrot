// Time for the heavy stuff
__global__ void mandelbrotKernel(int* d_output, int width, int height, 
                                 double center_x, double center_y, double zoom_scale, 
                                 int max_iter) {
    // global is function specifier in CUDA, it means called from cpu but ran on gpu

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // create variable idx

    if (idx >= width * height) return;
    // if thread is outside zone, stop                                    

    // Coordinate change
    int px = idx % width; // column
    int py = idx / width; // row

    // Scaling / Zooming logic
    double cr = (px - width / 2.0) * zoom_scale + center_x;
    double ci = (py - height / 2.0) * zoom_scale + center_y;

    // Mandelbrot iteration variables
    double zr = 0.0;
    double zi = 0.0;
    int iter = 0;

    // Run the loop
    while (iter < max_iter) {
        double zr2 = zr * zr;
        double zi2 = zi * zi;

        if (zr2 + zi2 > 4.0) {
            break; // escaped!
        }

        zi = 2.0 * zr * zi + ci;
        zr = zr2 - zi2 + cr;
        iter++;
    }

    d_output[idx] = iter;
}






// New stuff, this function is a bride, with it cpp can speak to gpu
void launchMandelbrot(int* d_pixels, int width, int height, 
                      double center_x, double center_y, double zoom_scale, int max_iter) {
// void means dont return
// int* d_pixels is a pointer to GPU memory

    int blockSize = 256;
    // this is just good size                    

    
    // Calculate how many blocks we need
    int numBlocks = (width * height + blockSize - 1) / blockSize;

    // EMBARASSINGLY PARELLEL
    // Call gpu and tell to get work on these stuff
    mandelbrotKernel<<<numBlocks, blockSize>>>(d_pixels, width, height, center_x, center_y, zoom_scale, max_iter);
    
    // wait for gpu to finnish
    cudaDeviceSynchronize();
}