// I'll still write some stuff up to help myself learn



// #include literally means copy file and paste it here

#include <iostream>
// Default c++ library which includes cout and cin
#include <fstream>
// Filestream, tools to make and write files
#include <cuda_runtime.h>
// This includes CUDA-functions like cudaMalloc, cudaMemcpy and cudaFree


// function to save image

// This function creates a simple PPM format picture
// PPM (portable pixel map) is a format which only has numbers after another
void savePGM(const int* data, int width, int height, const char* filename) {
    // void is the return type, "empty"
    // savePGM is the name, "portable gray map"
    // const means constant, a value that cannot be changed
    // int* is a pointer to the whole number table
    // int width, int height, size of image in pixels
    // const char* is a pointer to start of the characters



    std::ofstream file(filename);
    // std means standard namespace, default cpp library
    // ofstream, output filestream


    
    file << "P2\n" << width << " " << height << "\n255\n";
    // file << starts writing to file
    // this part is to make computer realize that PGM is not text but image
    // P2 means this is pgm, ASCII


    
    for (int i = 0; i < width * height; i++) {
        // start from 0, until all pixels are ran trough, i++ go to next pixel
        
        int pixel_value = data[i] % 255; 
        // searches the pixel value for i from data
        // % 255 is modulo, because pgm only understands 0-255
        
        file << pixel_value << " ";
        // writes the value to file

        if ((i + 1) % width == 0) file << "\n";
        // makes it easier to read for human eye
    }
    


    file.close();
    // closes the file writing pipe, frees the file

    std::cout << "Kuva tallennettu: " << filename << std::endl;
}   // Tell the user







// Time for the heavy stuff
__global__ void mandelbrotKernel(int* d_output, int width, int height, 
                                 float x_min, float x_max, float y_min, float y_max, 
                                 int max_iter) {
    // global is function specifier in CUDA, it means called from cpu but ran on gpu
    // void we know
    // gpu threads cannot return stuff normally, but rather write it to memory

    // int* d_outputs means it points into VRAM to data outputs, (which we will reserve with cudamalloc)
    // this is where each thread will write their value
    // those are just parameters which we will calculate
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // create variable idx
    // calculates global id for thread, not just group block etc

    

    if (idx >= width * height) return;
    // if thread is outside zone, stop                                    



    // Coordinate change
    int px = idx % width; // column
    int py = idx / width; // row, in Cpp / doesnt count decimals



    // Scaling
    // calculating "size" of pixel
    float dx = (x_max - x_min) / width;
    float dy = (y_max - y_min) / height;

    // calculating complexity
    float cr = x_min + px * dx; // real part
    float ci = y_min + py * dy; // imaginary part



    // Mandelbrot iteration
    // begin from zero, make the variables
    float zr = 0.0f;
    float zi = 0.0f;
    int iter = 0;



    // Run the loop until max iteration or |z| > 2, (mandelbrot)
    while (iter < max_iter) {


        // calculate exponent (faster than sqrt)
        float zr2 = zr * zr;
        float zi2 = zi * zi;

        // is |z| > 2 ?
        if (zr2 + zi2 > 4.0f) {
            break; // Karkasi!
        }

        

        // Calculating z_{n+1} = z_n^2 + c
        zi = 2.0f * zr * zi + ci;
        // the imaginary part (after opening brackets)
        zr = zr2 - zi2 + cr;
        // real

        iter++;
        // iter = iter + 1
    }



    d_output[idx] = iter;
    //now iter tells us how many iterations it survived, and saves it gpu, where idx points
}





// Time for the main stuff
int main() {
    // returns intege



    int width = 1024;
    int height = 1024;
    int max_iter = 1000;
    // Size of image, max interations

    // Defining the rectangle in complex space
    // this is typical mandelbrot rectangle
    float x_min = -2.5f;
    float x_max = 1.0f;
    float y_min = -1.2f;
    float y_max = 1.2f;



    // how much memory reserved
    size_t bytes = width * height * sizeof(int);
    // size_t "unsigned long long" 64 bit int with out sign in beginning
    // sizeof is a function which asks: "how many bytes one int is taking up"
    // calculates the size of one image (approx 4mb)



    // Allocation
    int* h_data = (int*)malloc(bytes); 
    // int* h_data, when main is running, it has a small very fast working memory called stack
    // h_data is a variable which is created to stack, it is a pointer
    // malloc(bytes) allocates memory from 64GB ram and returns the pointer to start
    // the (int*) tells the program that we are going to write only integers, also tells how to read the memory (int = 4 bytes)
    
    

    // create variable for device_data (neccesary for cuda shii)
    int* d_data;
    cudaMalloc(&d_data, bytes);
    // cuda malloc reserves bytes amount of data of my 32gb VRAM trough a PCIe-line       
    // &d_data tells the function, go write a pointer there



    // Kernel starting parameters
    int blockSize = 256;
    // threads per block
    
    // Calculate how many blocks we need to cover the image, rounds it up if neccesary
    int numBlocks = (width * height + blockSize - 1) / blockSize;

    std::cout << "Starting Mandelbrot-calculation with RTX 5090..." << std::endl;
    std::cout << "Resolution: " << width << "x" << height << std::endl;
    std::cout << "Number of threads: " << numBlocks * blockSize << std::endl;
    // std::endl; ends line and flushes



    // EMBARASSINGLY PARELLEL
    mandelbrotKernel<<<numBlocks, blockSize>>>(d_data, width, height, x_min, x_max, y_min, y_max, max_iter);
    // << >> is CUDA and means create numblocks groups with blocksize threads
    // then there's the parameters
    


    cudaDeviceSynchronize();
    // Stops CPU, to sync with gpu



    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    // This is cuda for transfering data, where, from where, how much, direction



    savePGM(h_data, width, height, "mandelbrot.pgm");
    // call for our helper function to save the data into the picture file


    // Free memories IMPORTANT
    cudaFree(d_data);
    free(h_data);

    return 0;
    // stops the shii
}