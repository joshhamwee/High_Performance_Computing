#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <mpi.h>

// Define output file name
#define OUTPUT_FILE "stencil.pgm"
#define MASTER 0

void stencil(int rank, const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image);
void stencil_final_column(int rank, const int final_column_nx, const int nx, const int ny, const int width, const int height,
            float* image, float* tmp_image);
void data_send_and_gather(const int rank, const int local_nx, const int final_column_nx, const int height,
            const int nprocs, const int tag, const int start, MPI_Status status, float*  image);
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

  //Determine rank to the left and right
  left = (rank == 0) ? (rank + nprocs - 1) : (rank - 1);
  right = (rank + 1) % nprocs;

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

  //Variables required for the stencil function
  int local_nx = floor(nx/nprocs); //Width of each column
  int final_column_nx = nx - (local_nx)*(nprocs-1); //Final column may be fifferent size
  int local_ny = height;
  int start = (1 + rank * local_nx) * local_ny; //Starting position of the top left pixel in each column

  //Creating the HALO Buffer
  float* HaloBuffer = malloc(local_ny*sizeof(float));

  //Create a derived data structure used for sending/receiving
  MPI_Datatype halo;
  MPI_Type_vector(local_ny, 1, 1, MPI_FLOAT, &halo);
  MPI_Type_commit(&halo);

  //Make sure all images have been initailised, wait here until they do
  MPI_Barrier(MPI_COMM_WORLD);

  double tic = wtime(); //Begin timing
  for (int t = 0; t < niters; ++t) {
    //Different stencil calls for final column and rest due to sizing
    if (rank == nprocs - 1) {
      stencil_final_column(rank, final_column_nx, local_nx, local_ny, width, height, image, tmp_image);
    }
    else{
      stencil(rank, local_nx, local_ny, width, height, image, tmp_image);
    }

    //HALO EXCHANGES
    if (rank == MASTER) {
      //Send RIGHT
      MPI_Send(&tmp_image[start + (local_nx-1)*local_ny], 1, halo, right, tag, MPI_COMM_WORLD);
      //Receive from RIGHT
      MPI_Recv(HaloBuffer, local_ny, MPI_FLOAT, right, tag, MPI_COMM_WORLD, &status);

      for (int y = 0; y < local_ny; y++) {
        tmp_image[start + (local_nx)*local_ny + y] = HaloBuffer[y];
      }
    }
    else if(rank == nprocs - 1){
      //Receive from LEFT
      MPI_Recv(HaloBuffer, local_ny, MPI_FLOAT, left, tag, MPI_COMM_WORLD, &status);
      //Send LEFT
      MPI_Send(&tmp_image[start], 1, halo, left, tag, MPI_COMM_WORLD);

      //Put in left
      for (int y = 0; y < local_ny; y++) {
        tmp_image[start - local_ny + y] = HaloBuffer[y];
      }
    }else{
      //Send right receive left
      MPI_Sendrecv(&tmp_image[start + (local_nx - 1)*local_ny], 1, halo, right, tag,HaloBuffer, local_ny, MPI_FLOAT, left, tag, MPI_COMM_WORLD, &status);

      //Put in left
      for (int y = 0; y < local_ny; y++) {
        tmp_image[start - local_ny + y] = HaloBuffer[y];
      }
      //Send left receive right
      MPI_Sendrecv(&tmp_image[start], 1, halo, left, tag,HaloBuffer, local_ny, MPI_FLOAT, right, tag, MPI_COMM_WORLD, &status);

      //Put in right
      for (int y = 0; y < local_ny; y++) {
        tmp_image[start + local_nx*local_ny + y] = HaloBuffer[y];
      }
    }

    //Call stencil again with images tmp_image and image rotated
    if (rank == nprocs - 1) {
      stencil_final_column(rank, final_column_nx, local_nx, local_ny, width, height, tmp_image, image);
    }
    else{
      stencil(rank, local_nx, local_ny, width, height, tmp_image, image);
    }

    //HALO EXCHANGES
    if (rank == MASTER) {
      //Send RIGHT
      MPI_Send(&image[start + (local_nx-1)*local_ny], 1, halo, right, tag, MPI_COMM_WORLD);
      //Receive from RIGHT
      MPI_Recv(HaloBuffer, local_ny, MPI_FLOAT, right, tag, MPI_COMM_WORLD, &status);

      for (int y = 0; y < local_ny; y++) {
      image[start + (local_nx)*local_ny + y] = HaloBuffer[y];
      }
    }
    else if(rank == nprocs - 1){
      //Receive from LEFT
      MPI_Recv(HaloBuffer, local_ny, MPI_FLOAT, left, tag, MPI_COMM_WORLD, &status);
      //Send LEFT
      MPI_Send(&image[start], 1, halo, left, tag, MPI_COMM_WORLD);

      //Put from LEFT
      for (int y = 0; y < local_ny; y++) {
        image[start - local_ny + y] = HaloBuffer[y];
      }
    }else{
      //Send right receive left
      MPI_Sendrecv(&image[start + (local_nx - 1)*local_ny], 1, halo, right, tag,HaloBuffer, local_ny, MPI_FLOAT, left, tag, MPI_COMM_WORLD, &status);

      //Put in left
      for (int y = 0; y < local_ny; y++) {
        image[start - local_ny + y] = HaloBuffer[y];
      }
      //Send left receive right
      MPI_Sendrecv(&image[start], 1, halo, left, tag,HaloBuffer, local_ny, MPI_FLOAT, right, tag, MPI_COMM_WORLD, &status);

      //Put in right
      for (int y = 0; y < local_ny; y++) {
        image[start + local_nx*local_ny + y] = HaloBuffer[y];
      }
    }
  }
  double toc = wtime(); //End timing

  //Collate all the columns and merge into master
  data_send_and_gather(rank, local_nx, final_column_nx, height, nprocs, tag, start, status, image);


  // Output
  printf("------------------------------------\n");
  printf(" runtime: %lf s\n", toc - tic);
  printf("------------------------------------\n");

  if(rank == MASTER){
    output_image(OUTPUT_FILE, nx, ny, width, height, image);
  }

  free(image);
  free(tmp_image);

  MPI_Finalize();
}

void stencil(int rank, const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image)
{
  for (int j = 1; j < ny - 1; ++j) {
    for (int i = 1 + nx*rank; i < (nx + 1) + (nx*rank); ++i) {
      tmp_image[j + i * height] =  image[j     + i       * height] * 0.6f;
      tmp_image[j + i * height] += (image[j     + (i - 1) * height] + image[j     + (i + 1) * height] + image[j - 1 + i       * height] + image[j + 1 + i       * height])* 0.1f;
    }
  }
}

void stencil_final_column(int rank, const int final_column_nx, const int nx, const int ny, const int width, const int height,
             float* image, float* tmp_image)
{
  for (int j = 1; j < ny - 1; ++j) {
    for (int i = 1 + nx*rank; i < (final_column_nx + (1 + nx*rank)); ++i) {
      tmp_image[j + i * height] =  image[j     + i       * height] * 0.6f;
      tmp_image[j + i * height] += (image[j     + (i - 1) * height] + image[j     + (i + 1) * height] + image[j - 1 + i       * height] + image[j + 1 + i       * height])* 0.1f;
    }
  }
}

//After stencil function finished send all data back to master
void data_send_and_gather(const int rank, const int local_nx, const int final_column_nx, const int height,
   const int nprocs, const int tag, const int start, MPI_Status status, float* image){

     float senderbuffer[(local_nx*height)]; //Buffer to be sent if not final column
     float senderbuffer_finalcolumn[((final_column_nx)*height)]; //Final column may be different size to others so need different buffer

     if(rank == MASTER){
       //Receive all column buffers from columns 1 -> 1 - final columns
       for (int nreceive = 1; nreceive < nprocs - 1; nreceive++) {
         MPI_Recv(senderbuffer, local_nx*height, MPI_FLOAT, nreceive, tag, MPI_COMM_WORLD, &status);
         for (int i = 0; i < (local_nx*height); i++) {
           image[((local_nx*nreceive)+1)*height + i] = senderbuffer[i];
         }
       }
       //Final column man be different size so receive from different buffer
       MPI_Recv(senderbuffer_finalcolumn, (final_column_nx)*height, MPI_FLOAT, (nprocs-1), tag, MPI_COMM_WORLD, &status);
       for (int i = 0; i < ((final_column_nx)*height); i++) {
         image[((local_nx*(nprocs-1))+1)*height + i] = senderbuffer_finalcolumn[i];
       }
     }
     //Send final column to master
     else if (rank == nprocs - 1) {
       for (int i = 0; i < ((final_column_nx)*height); i++) {
         senderbuffer_finalcolumn[i] = image[start + i];
       }
       MPI_Send(senderbuffer_finalcolumn, (final_column_nx)*height, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
     }
     //Send all other columns to master
     else{
       for (int i = 0; i < (local_nx*height); i++) {
         senderbuffer[i] = image[start + i];
       }
       MPI_Send(senderbuffer, local_nx*height, MPI_FLOAT, MASTER, tag, MPI_COMM_WORLD);
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
