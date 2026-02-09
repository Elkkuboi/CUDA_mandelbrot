#ifndef MANDELBROT_H
#define MANDELBROT_H

#include <cstdint> 

void launchMandelbrot(int* d_pixels, int width, int height, 
                      double center_x, double center_y, double zoom_scale, int max_iter);

#endif

// this file just tells the cpu that gpu has this kind of a function