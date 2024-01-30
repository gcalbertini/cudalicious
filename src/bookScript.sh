# Compile CUDA file
nvcc -c -arch=sm_70 -o vectorAdd.o vectorAdd.cu

# Compile main file
#g++ -o main.o -c main.cpp

# Link the object files and execute (add main.o next too xyz.o if you need those too)
nvcc -arch=sm_70 -o my_program vectorAdd.o
./my_program
