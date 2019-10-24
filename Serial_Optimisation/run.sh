##
rm stencil
rm stencil.out
rm stencil.pgm
icc -fast -O3 -xHOST -std=c99 stencil.c -o stencil 
sbatch stencil.job
sleep 20
python check.py --ref-stencil-file stencil_8000_8000_100.pgm --stencil-file stencil.pgm
cat stencil.out

