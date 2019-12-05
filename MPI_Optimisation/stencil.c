#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(const int start, const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image);
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image);
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image);
double wtime(void);


int main(int argc, char* argv[])
{
  // Check usage
  if (argc != 4) {
    fprintf(stderr, "Usage: %s nx ny niters\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  //MPI Startup
  MPI_Init(&argc, &argv);
  //Number processors, which processor ...
  int nprocs, rank, flag, left, right, tag = 0;
  MPI_Status status;

  MPI_Initialized(&flag);
  if ( flag != 1 ){
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }

  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &nprocs );

  printf("Hello from rank %d of %d\n", rank, nprocs);

  //Determine rank to the left and right
  left = (rank == 0) ? (rank + nprocs - 1) : (rank - 1);
  right = (rank + 1) % nprocs;

  printf("left is %d and right is %d\n", left, right);

  // Initiliase problem dimensions from command line arguments
  int nx = atoi(argv[1]);
  int ny = atoi(argv[2]);
  int niters = atoi(argv[3]);

  // we pad the outer edge of the image to avoid out of range address issues in
  // stencil

  int width = nx + 2;
  int height = ny + 2;

  // Allocate the image
  float* image = malloc(sizeof(float) * width * height);
  float* tmp_image = malloc(sizeof(double) * width * height);

  // Set the input image
  init_image(nx, ny, width, height, image, tmp_image);

  int local_nx = (nx / nprocs) + 1;
  int local_ny = ny;

  int local_width = (width / 2);
  int local_height = height;

  int start;
  if (rank == 0) {
    start = 0;
  }else{
    start = 512;
  }

  MPI_Datatype halo;
  float* haloN = malloc(ny*sizeof(float));

  MPI_Type_vector(ny, 1, 1, MPI_FLOAT, &halo);
  MPI_Type_commit(&halo);

  // syncronise processes
  MPI_Barrier(MPI_COMM_WORLD);

  // Call the stencil kernel
  double tic = wtime();
  for (int t = 0; t < niters; ++t) {
    //TODO figure out halo send and receives
    stencil(start, local_nx, local_ny, local_width, local_height, image, tmp_image);
    if (rank == 0) {
      //Send start + local_nx - 1;
      MPI_Send(&tmp_image[start+(local_nx-1)*local_ny], 1, halo, right, tag, MPI_COMM_WORLD);
      //Receive start + local_nx;
      MPI_Recv(haloN, ny, MPI_FLOAT, right, tag, MPI_COMM_WORLD, &status);

      for (int y = 0; y < ny; y++) {
        tmp_image[start + (local_nx-1)*ny + y] = haloN[y];
      }

    }else{
      MPI_Recv(haloN, ny, MPI_FLOAT, left, tag, MPI_COMM_WORLD, &status);
      //Send start + 1;
      MPI_Send(&tmp_image[start+(1)*ny], 1, halo, left, tag, MPI_COMM_WORLD);
      for (int y = 0; y < ny; y++) {
        tmp_image[start + y] = haloN[y];
      }
    }
    stencil(start, local_nx, local_ny, local_width, local_height, tmp_image, image);
    if (rank == 0) {
      //Send start + local_nx - 1;
      MPI_Send(&image[start+(local_nx-1)*local_ny], 1, halo, right, tag, MPI_COMM_WORLD);
      //Receive start + local_nx;
      MPI_Recv(haloN, ny, MPI_FLOAT, right, tag, MPI_COMM_WORLD, &status);

      for (int y = 0; y < ny; y++) {
        image[start + (local_nx-1)*ny + y] = haloN[y];
      }

    }else{
      MPI_Recv(haloN, ny, MPI_FLOAT, left, tag, MPI_COMM_WORLD, &status);
      //Send start + 1;
      MPI_Send(&image[start+(1)*ny], 1, halo, left, tag, MPI_COMM_WORLD);
      for (int y = 0; y < ny; y++) {
        image[start + y] = haloN[y];
      }
    }
  }
  double toc = wtime();
  float senderbuffer[(512*1024)];
  if(rank == MASTER){
    MPI_Recv(senderbuffer, 512*1024, MPI_FLOAT, 1, tag, MPI_COMM_WORLD, &status);
    for (int i = 0; i < (512*1024); i++) {
      image[1 + 512 + i] = senderbuffer[i];
    }
  }
  else{
    for (int i = 0; i < (512*1024); i++) {
      senderbuffer[i] = image[1 + start + i];
    }
    MPI_Send(senderbuffer, 512*1024, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
  }

  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc - tic);
  printf("------------------------------------\n");

  output_image(OUTPUT_FILE, nx, ny, width, height, image);
  free(image);
  free(tmp_image);

  MPI_Finalize();
}

void stencil(const int start, const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image)
{
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1 + start; i < start + nx + 1; ++i) {
      tmp_image[j + i * height] =  image[j     + i       * height] * 0.6f;
      tmp_image[j + i * height] += (image[j     + (i - 1) * height] + image[j     + (i + 1) * height] + image[j - 1 + i       * height] + image[j + 1 + i       * height])* 0.1f;
    }
  }
}

// Create the input image
void init_image(const int nx, const int ny, const int width, const int height,
                float* image, float* tmp_image)
{
  // Zero everything
  for (int j = 0; j < ny + 2; ++j) {
    for (int i = 0; i < nx + 2; ++i) {
      image[j + i * height] = 0.0;
      tmp_image[j + i * height] = 0.0;
    }
  }

  const int tile_size = 64;
  // checkerboard pattern
  for (int jb = 0; jb < ny; jb += tile_size) {
    for (int ib = 0; ib < nx; ib += tile_size) {
      if ((ib + jb) % (tile_size * 2)) {
        const int jlim = (jb + tile_size > ny) ? ny : jb + tile_size;
        const int ilim = (ib + tile_size > nx) ? nx : ib + tile_size;
        for (int j = jb + 1; j < jlim + 1; ++j) {
          for (int i = ib + 1; i < ilim + 1; ++i) {
            image[j + i * height] = 100.0;
          }
        }
      }
    }
  }
}

// Routine to output the image in Netpbm grayscale binary image format
void output_image(const char* file_name, const int nx, const int ny,
                  const int width, const int height, float* image)
{
  // Open output file
  FILE* fp = fopen(file_name, "w");
  if (!fp) {
    fprintf(stderr, "Error: Could not open %s\n", OUTPUT_FILE);
    exit(EXIT_FAILURE);
  }

  // Ouptut image header
  fprintf(fp, "P5 %d %d 255\n", nx, ny);

  // Calculate maximum value of image
  // This is used to rescale the values
  // to a range of 0-255 for output
  double maximum = 0.0;
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      if (image[j + i * height] > maximum) maximum = image[j + i * height];
    }
  }

  // Output image, converting to numbers 0-255
  for (int j = 1; j < ny + 1; ++j) {
    for (int i = 1; i < nx + 1; ++i) {
      fputc((char)(255.0 * image[j + i * height] / maximum), fp);
    }
  }

  // Close the file
  fclose(fp);
}

// Get the current time in seconds since the Epoch
double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}
