// Homework 1
// Color to Greyscale Conversion

// A common way to represent color images is known as RGBA - the color
// is specified by how much Red, Grean and Blue is in it.
// The 'A' stands for Alpha and is used for transparency, it will be
// ignored in this homework.

// Each channel Red, Blue, Green and Alpha is represented by one byte.
// Since we are using one byte for each color there are 256 different
// possible values for each color.  This means we use 4 bytes per pixel.

// Greyscale images are represented by a single intensity value per pixel
// which is one byte in size.

// To convert an image from color to grayscale one simple method is to
// set the intensity to the average of the RGB channels.  But we will
// use a more sophisticated method that takes into account how the eye
// perceives color and weights the channels unequally.

// The eye responds most strongly to green followed by red and then blue.
// The NTSC (National Television System Committee) recommends the following
// formula for color to greyscale conversion:

// I = .299f * R + .587f * G + .114f * B

// Notice the trailing f's on the numbers which indicate that they are
// single precision floating point constants and not double precision
// constants.

// You should fill in the kernel as well as set the block and grid sizes
// so that the entire image is processed.

#include "utils.h"

__global__ void rgba_to_greyscale(const uchar4 *const rgbaImage,
                                  unsigned char *const greyImage,
                                  int numRows, int numCols)
{
  // TODO
  // Fill in the kernel to convert from color to greyscale
  // the mapping from components of a uchar4 to RGBA is:
  //  .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  // The output (greyImage) at each pixel should be the result of
  // applying the formula: output = .299f * R + .587f * G + .114f * B;
  // Note: We will be ignoring the alpha channel for this conversion

  // First create a mapping from the 2D block and grid locations
  // to an absolute 2D location in the image, then use that to
  // calculate a 1D offset

  /*
    The index of a thread and its thread ID relate to each other in a straightforward way:
    For a one-dimensional block, they are the same; for a two-dimensional block of size (Dx, Dy),
    the thread ID of a thread of index (x, y) is (x + y Dx); for a three-dimensional block of size (Dx, Dy, Dz),
    the thread ID of a thread of index (x, y, z) is (x + y Dx + z Dx Dy).
  */

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  // Is thread within the image bounds?
  if (row < numRows && col < numCols)
  {
    // 1D offset for the current pixel
    int offset = col * numRows + row;
    const uchar4 *shifted = rgbaImage + offset;

    unsigned char red = shifted->x;
    unsigned char green = shifted->y;
    unsigned char blue = shifted->z;

    greyImage[offset] = static_cast<unsigned char>(0.299f * red + 0.587f * green + 0.114f * blue);
  }
}

void your_rgba_to_greyscale(const uchar4 *const h_rgbaImage, uchar4 *const d_rgbaImage,
                            unsigned char *const d_greyImage, size_t numRows, size_t numCols)
{
  // You must fill in the correct sizes for the blockSize and gridSize
  // currently only one block with one thread is being launched
  const dim3 blockSize(16, 16, 1); // Set 16x16 to start

  // if the input dimensions are not exact multiples of the block size, adjust the grid dimensions to cover the entire input
  int gridSizeX = numCols + (16 - (numCols % 16));
  int gridSizeY = numRows + (16 - (numRows % 16));
  const dim3 gridSize(gridSizeX, gridSizeY, 1);
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}
