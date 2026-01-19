// I'll write some stuff up to help myself learn, dont mind it




// #include literally means copy file and paste it here
#include <iostream> 
// Default c++ library which includes cout and cin
#include <cuda_runtime.h>
// This includes CUDA-functions like cudaMalloc, cudaMemcpy and cudaFree



// This is a gpu function (kernel)
// It will not yet calculate mandelbrot, just test the memory


// global tells to execute with gpu but called with cpu
// void means no return type
// float is a float, * means a pointer, it will not include the number but an adress to memory
// d_data is a variable name, d_ means device which reminds it will be on gpu
// int_n will tell how many numbers are there in the table

__global__ void testKernel(float* d_data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // int idx creates a variable for index
    // threadIdx.x is "my number" in this group meaning if there's 32 threads in the group this will be a number between 0-31
    // blockIdx.x is the number of the group, we only have one group so this will be 0
    // blockDim.x is the size of one group

    // So the equation (group number * group size) + my number in group is unique global id for each thread in the whole gpu

    if (idx < n) {
        // To check if we accidentally power 1000 threads but fit only 32, 33-999 would be tried to be writen outside memory, not good
        d_data[idx] = (float)idx; 
        // Go to d_data idx, change the value to idx which is float type
    }
}

// int return 0 to OS if all went well, else there is an error
int main() {
    int n = 32;
    size_t bytes = n * sizeof(float);
    // size_t "Size type" is a special kind of integer, which is decided to represent memory size, it is positive and big enough to cover RAM
    // sizeof(float) is a function, which tells us how many bytes does one float take, usually 4, so bytes would be 32 * 4 = 128

    // Reserve memory from host (CPU)
    float* h_data = (float*)malloc(bytes);
    // create a pointer named h_data (host_data)
    // malloc(bytes) Memory ALLOCation. Reserves 128 bytes (or whatever) from memory, returns the pointer to the beginning of the 128 bytes
    // malloc by itself returns a "generic pointer" (void*) which means it doesn't know what will be stored there, therefore we use (float*)


    // Create pointer d_data (device_data) atp it is an empty pointer in CPU memory, which *can* include an adress.
    float* d_data;

    cudaMalloc(&d_data, bytes);
    // cudaMalloc is nvidias malloc, it reserves space from VRAM (which I have 32gb of *skull*) BUT: it return error coding not the adress
    // & means "adress of". We want cudamalloc to chagne d_data pointer number, right now it points to nothing or garbage
    // If we gave it d_data, we would give it the garbage it points at, now we ar telling it; hey, there's the pointer (d_data) we want to change
    // we want cudamalloc to go to d_data and change it

    // <<< ... >>> is a CUDA expansion, first number is grid size, and second number is block size (how many threads)
    // d_data and n are the arguments. NOTE, we are giving it d_data (gpu-pointer) not h_data because GPU could not access CPU's 
    // testKernel is the function we created
    // it will go to d_data idx and change value to the equation we did (float)
    testKernel<<<1, 32>>>(d_data, n);


    // Copying the result back to CPU
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    // cudaMemcpy means transfering data trough PCIe, h_data is the pointer to destination, 
    // d_data is the pointer of source, bytes is how much should we copy
    // cudaMemcpyDeviceToHost just means direction, from device to host



    // 5. Checker
    std::cout << "GPU wrote to memory: ";
    // character output "print", << what to cout

    for (int i = 0; i < 10; i++) { // print first 10
        std::cout << h_data[i] << " "; // read data from cpu memory from our h_data slot, which we had reserved and trasnfered the data to
    }
    std::cout << "..." << std::endl;

    // Cleaning our VRAM
    cudaFree(d_data);
    // frees what we save, actually it knows how much to free from secret metadata which it saves (how many bytes)
    free(h_data);

    return 0;
    // 0 means good
}