#include <SDL2/SDL.h>
// Simple directmedia layer 2
// Portable access to windowing, input, audio and simple graphics
#include <iostream>
// Default c++ library which includes cout and cin
#include <cuda_runtime.h>
// This includes CUDA-functions like cudaMalloc, cudaMemcpy and cudaFree
#include <cmath>
// sin
#include "mandelbrot.h"
// our own header

// window size
const int WIDTH = 1024;
const int HEIGHT = 1024;

int main(int argc, char* argv[]) {
    // returns 0 if success, -1 if failure
    // int argc means argument count, how many words were typed when program started
    // char* argv[] means argument vector, which is a table of the words
    // we really dont use these but standard way of making stuff



    // SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) return -1;
    // SDL_init runs SDL librarty, we ask SDL only to turn on video-properties



    SDL_Window* window = SDL_CreateWindow("RTX 5090 Mandelbrot - Mouse Zoom", 
                                          SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 
                                          WIDTH, HEIGHT, SDL_WINDOW_SHOWN);
        // SDL_Window* is a pointer to window object, this is the frame
        // SDL_CreateWindow asks OS to open a window, and then just simple stuff



    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        // SDL_Renderer* is a pointer to the renderer object, which makes sure pixel go inside the window, -1 is first gpu driver



    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, 
                                             SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);
        // SDL_Texture* pointer to texture object, this is the place in memory where it writes the picutre and which render then
        // takes and places into window. Streaming means we are going to change constantly



    // ram and vram
    size_t data_size = WIDTH * HEIGHT * sizeof(int);
        // Size type positive int to tell size, sizeof tells how many bytes one int takes



    int* d_output;
    cudaMalloc(&d_output, data_size);
        // int* is a pointer to an integer, device_output
        // cudamalloc allocates memory and &d_output "write allocated memory address to this variable"
        


    int* h_output = new int[WIDTH * HEIGHT];
    uint32_t* screen_pixels = new uint32_t[WIDTH * HEIGHT];
        // host_device new reserves memory from RAM
        // uint32_t is unsigned integer 32-bit, it means always 32 bit int, good for colors



    // SETUP CAMERA AND SPACE
    double centerX = -0.75; 
    double centerY = 0.0;
    double zoomScale = 0.004;
    int maxIter = 500;

    bool running = true;
    SDL_Event event;
    // running keeps running while true
    // SDL_event is where linux tells what is going on
    // mailbox


    // GAME LOOP
    while (running) {
        


        // input handling
        while (SDL_PollEvent(&event)) {
            // is there mail in the box?
            if (event.type == SDL_QUIT) running = false;
            if (event.type == SDL_KEYDOWN) {
                if (event.key.keysym.sym == SDLK_ESCAPE) running = false;
                // Tarkkuuden säätö lennossa
                // if (event.key.keysym.sym == SDLK_e) maxIter += 100; 
                // if (event.key.keysym.sym == SDLK_q) maxIter -= 100;
            }
        }




        // The mouse shii, this shit is hard 
        int mouseX, mouseY;
        uint32_t mouseButtons = SDL_GetMouseState(&mouseX, &mouseY);
        // getmousestate return what is pressed, puts the mouse coordinates in mouseX and mouseY



        // Zoom if pressed
        if (mouseButtons & SDL_BUTTON(SDL_BUTTON_LEFT) || mouseButtons & SDL_BUTTON(SDL_BUTTON_RIGHT)) {
            


            // calculate location of mouse
            double mouseRe = (mouseX - WIDTH / 2.0) * zoomScale + centerX;
            double mouseIm = (mouseY - HEIGHT / 2.0) * zoomScale + centerY;



            // zooming in
            if (mouseButtons & SDL_BUTTON(SDL_BUTTON_LEFT)) {
                zoomScale *= 0.99; // change zoomscale
            } else {
                zoomScale *= 1.01; // zoom out
            }



            // update center
            // we want the center to be the same after the zoom also
            // now zooming happends where the mouse is and not in the center (black)
            centerX = mouseRe - (mouseX - WIDTH / 2.0) * zoomScale;
            centerY = mouseIm - (mouseY - HEIGHT / 2.0) * zoomScale;
        }



        // Calculation and drawing
        launchMandelbrot(d_output, WIDTH, HEIGHT, centerX, centerY, zoomScale, maxIter);
        cudaMemcpy(h_output, d_output, data_size, cudaMemcpyDeviceToHost);
        // cudamemphy is a function to copy memory between devices



        // coloring
        for (int i = 0; i < WIDTH * HEIGHT; i++) {
            // go trough every pixel



            int iter = h_output[i];
            // find iteration
            uint32_t color;
            // variable



            if (iter >= maxIter) {
                color = 0xFF000000; 
                // black if outside

                
                
            } else {
                double t = (double)iter * 0.01;
                // make iter float
                // 0.01 is just how fast colors change



                // this is just different parts of sine waves
                int r = (int)((sin(t + 0.0) * 127 + 128));
                int g = (int)((sin(t + 2.0) * 127 + 128));
                int b = (int)((sin(t + 4.0) * 127 + 128));

                color = (255 << 24) | (r << 16) | (g << 8) | b;
                // packaging all information to one 32byte shii


            }
            screen_pixels[i] = color;
            // save the color to table
        }



        SDL_UpdateTexture(texture, NULL, screen_pixels, WIDTH * sizeof(uint32_t));
        // send screen_pixels table from ram to texture, (we are telling its size)
        SDL_RenderClear(renderer);
        // wipe the shii
        SDL_RenderCopy(renderer, texture, NULL, NULL);
        // copy the new one to driver, null means use whole image, another null means widen it to image
        SDL_RenderPresent(renderer);
        // don't write straight to screen (Vsync)
    }

    // Cleaning
    cudaFree(d_output);
    delete[] h_output;
    delete[] screen_pixels;
    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}