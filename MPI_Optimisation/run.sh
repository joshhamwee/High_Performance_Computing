##
rm stencil
rm stencil.out
rm stencil.pgm
mpicc -std=c99 stencil.c -o stencil
sbatch stencil.job
sleep 20
python check.py --ref-stencil-file stencil_1024_1024_100.pgm --stencil-file stencil.pgm
cat stencil.out
