![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document (Day 2)

2024-12-03-ds-gpu

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------


##  🫱🏽‍🫲🏻 Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop website

* [Workshop page](https://esciencecenter-digital-skills.github.io/2024-12-03-ds-gpu/)
* [Google Colab](https://colab.research.google.com/)

## 👩‍🏫👩‍💻🎓 Instructors

Alessio Sclocco, Johan Hidding, Laurent Soucasse

## 🧑‍🙋 Helpers

## 🗓️ Agenda
|  Time | Topic                            |
| -----:|:-------------------------------- |
| 09:30 | Welcome and icebreaker           |
| 09:45 | Introduction to CUDA             |
| 10:30 | Coffee break                     |
| 10:45 | Introduction to CUDA             |
| 11:15 | Coffee break                     |
| 11:30 | Introduction to CUDA             |
| 12:00 | Lunch break                      |
| 13:00 | CUDA memories and their use      |
| 14:00 | Coffee break                     |
| 14:15 | Data sharing and synchronization |
| 15:00 | Coffee break                     |
| 15:15 | Concurrent access to the GPU     |
| 16:15 | Wrap-up                          |
| 16:30 | Drinks                           |
| 17:00 | END                              |


## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, 
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl.
Please request your certificate within 8 months after the workshop, as we will delete all personal identifyable information after this period.

## 🔧 Exercises

#### Increase vector size

Modify our CUDA Kernel to sum vectors of bigger size. The solution is below. We modified the Kernel such that we use the threads of two different blocks.

```python
import cupy as cp

size = 2048

a_gpu = cp.random.rand(size, dtype=cp.float32)
b_gpu = cp.random.rand(size, dtype=cp.float32)
c_gpu = cp.zeros(size, dtype=cp.float32)

vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float* A, const float* B, float* C, const int size) {
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   C[item] = A[item] + B[item];
}
'''

vector_add_gpu = cp.RawKernel(vector_add_cuda_code, "vector_add")

vector_add_gpu((2, 1, 1), (size // 2, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```
#### Scale up

The code above work but it is too much specific of the vector size and we want to be able to address whatever input size. We need to split the total work (the input size of our vector) accross all the blocks with limited size of 1024. Below, we parametrize the CUDA grid size (`gridDim`) such that we are able to address input vector of arbitrary size. We also need to check inside our kernel if a thread is not trying to address items out of bounds of the input vector.
```python
import cupy as cp
import math

size = 10000
threads_per_block = 1024

a_gpu = cp.random.rand(size, dtype=cp.float32)
b_gpu = cp.random.rand(size, dtype=cp.float32)
c_gpu = cp.zeros(size, dtype=cp.float32)

vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   if ( item < size )
   {
      C[item] = A[item] + B[item];
   }
}
'''

vector_add_gpu = cp.RawKernel(vector_add_cuda_code, "vector_add")

grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size))
```
#### New challenge compute prime numbers using CUDA
We come back to the example of the previous episode where we want to compute prime numbers on the GPU, this time using your own CUDA kernel. You will find below a template which provides correct output but has to be fixed to benefit from the GPU computational power.
```python
import numpy as np
import cupy
import math
from cupyx.profiler import benchmark

# CPU version
def all_primes_to(upper : int, prime_list : list):
    for num in range(0, upper):
        prime = True
        if num == 1:
            prime = False
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                prime = False
                break
        if prime:
            prime_list[num] = 1

upper_bound = 100_000
all_primes_cpu = np.zeros(upper_bound, dtype=np.int32)

# GPU version
check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
   for ( int number = 0; number < size; number++ )
   {
       int result = 1;
       if ( number == 1 )
       {
           result = 0;
       }  
       for ( int factor = 2; factor <= number / 2; factor++ )
       {
           if ( number % factor == 0 )
           {
               result = 0;
               break;
           }
       }

       all_prime_numbers[number] = result;
   }
}
'''
# Allocate memory
all_primes_gpu = cupy.zeros(upper_bound, dtype=cupy.int32)

# Setup the grid
all_primes_to_gpu = cupy.RawKernel(check_prime_gpu_code, "all_primes_to")
grid_size = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size = (1024, 1, 1)

# Benchmark and test
%timeit -n 1 -r 1 all_primes_to(upper_bound, all_primes_cpu)
execution_gpu = benchmark(all_primes_to_gpu, (grid_size, block_size, (upper_bound, all_primes_gpu)), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s")

if np.allclose(all_primes_cpu, all_primes_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
```

#### Our solution
Here is how you can modify the CUDA kernel to benefit from the GPU. Instead of looping over the elements of our input vector (`number`), we assign the work on each element to a different thread of the GPU.

```cpp
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
    int number = (blockIdx.x * blockDim.x) + threadIdx.x;
    int result = 1;

    if ( number < size )
    {
        if ( number == 1 )
        {
                result = 0;
        }     
        for ( int factor = 2; factor <= number / 2; factor++ )
        {

            if ( number % factor == 0 )
            {
                result = 0;
                break;
            }
        }

        all_prime_numbers[number] = result;
    }
}
```

### Challenge: parallel reduction

Modify the parallel reduction CUDA kernel and make it work.

Kernel:

```python
cuda_code = r'''
#define block_size_x 256

extern "C"
__global__ void reduce_kernel(float *out_array, float *in_array, int n) {

    int ti = threadIdx.x;
    int x = blockIdx.x * block_size_x + threadIdx.x;
    int step_size = gridDim.x * block_size_x;
    float sum = 0.0f;

    //cooperatively (with all threads in all thread blocks) iterate over input array
    for (int i=x; i<n; i+=step_size) {
        sum += in_array[i];
    }

    //at this point we have reduced the number of values to be summed from n to
    //the total number of threads in all thread blocks combined

    //the goal is now to reduce the values within each thread block to a single
    //value per thread block for this we will need shared memory

    //declare shared memory array, how much shared memory do we need?
    __shared__ float temp_reduction[block_size_x];

    //make every thread store its thread-local sum to the array in shared memory
    temp_reduction[threadIdx.x] = sum;
    
    //now let's call syncthreads() to make sure all threads have finished
    //storing their local sums to shared memory
    __syncthreads();

    //now this interesting looking loop will do the following:
    //it iterates over the block_size_x with the following values for s:
    //if block_size_x is 256, 's' will be powers of 2 from 128, 64, 32, down to 1.
    //these decreasing offsets can be used to reduce the number
    //of values within the thread block in only a few steps.
    #pragma unroll
    for (unsigned int s=block_size_x/2; s>0; s/=2) {

        //you are to write the code inside this loop such that
        //threads will add the sums of other threads that are 's' away
        //do this iteratively such that together the threads compute the
        //sum of all thread-local sums 

        //use shared memory to access the values of other threads
        //and store the new value in shared memory to be used in the next round
        //be careful that values that should be read are
        //not overwritten before they are read
        //make sure to call __syncthreads() when needed
        
        if (ti < s) {
           temp_reduction[ti] += temp_reduction[ti + s];
        }
        __syncthreads();
    }

    //write back one value per thread block
    if (ti == 0) {
        out_array[blockIdx.x] = temp_reduction[0];  //store the per-thread block reduced value to global memory
    }
}
'''
```

Python host code, it does not need to be modified for the exercise.

```python
import numpy
import cupy

# Allocate memory
size = numpy.int32(5e7)
input_cpu = numpy.random.randn(size).astype(numpy.float32) + 0.00000001
input_gpu = cupy.asarray(input_cpu)
out_gpu = cupy.zeros(2048, dtype=cupy.float32)

# Compile CUDA kernel
grid_size = (2048, 1, 1)
block_size = (256, 1, 1)
reduction_gpu = cupy.RawKernel(cuda_code, "reduce_kernel")

# Execute athe first partial reduction
reduction_gpu(grid_size, block_size, (out_gpu, input_gpu, size))
# Execute the second and final reduction
reduction_gpu((1, 1, 1), block_size, (out_gpu, out_gpu, 2048))

# Execute and time CPU code
sum_cpu = numpy.sum(input_cpu)

if numpy.absolute(sum_cpu - out_gpu[0]) < 1.0:
    print("Correct results!")
else:
    print("Wrong results!")
```

## 🧠 Collaborative Notes

### First CUDA GPU Kernel
We write our first CUDA Kernel to sum two vectors.
```python
import cupy as cp

size = 1024

a_gpu = cp.random.rand(size, dtype=cp.float32)
b_gpu = cp.random.rand(size, dtype=cp.float32)
c_gpu = cp.zeros(size, dtype=cp.float32)

vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float* A, const float* B, float* C, const int size) {
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
'''

vector_add_gpu = cp.RawKernel(vector_add_cuda_code, "vector_add")

vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```
The `threadIdx` variable is a 3D index of a thread on the GPU. 3D comes from imaging purposes. You can access the components using `threadIdx.x` (x, y, z). Threads are grouped by blocks, you can access the numbers of threads in the blocks using `blockDim`, the index of each block with `blockIdx` and the number of blocks using `gridDim` (all triplets). In the `vector_add_gpu` call, the first triplet we specify corresponds to the size of the CUDA grid (`gridDim`, here made of one block for each dimension) and the second triplet corresponds to the size of the blocks (`blockDim`).

```python
#CPU implementation of vector add to check
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]
        
import numpy as np

a_cpu = cp.asnumpy(a_gpu)
b_cpu = cp.asnumpy(b_gpu)
c_cpu = np.zeros(size, dtype=np.float32)

vector_add(a_cpu, b_cpu, c_cpu, size)

# test
if np.allclose(c_cpu, c_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
```
If we now want to increase the size of our input, let's say `size = 2048`, we get a CUDA error as usually blocks contain a maximum of 1024 threads => see first challenge!

### GPU hardware and memories
There are three levels on the GPU (hierarchy) we previously discussed: the grid > the block > the thread.
- **Global memory** is defined at the grid level and is accessible by all the threads (shared memory) to read and write. It is allocated by the host (the CPU). In addition, the host can allocate shared but read-only memory called **constant memory**, with faster access. This memory needs to be declared outside the kernel with the `__constant__` argument. An example use case is to store the filter to be used in a convolution process. Finally, we find at the grid level the **texture memory** which is a specific type of memory for rendering 3D images.
- At the block level, the device (the GPU) can allocate block-specific memory called **shared memomy**, i.e. an amount of memomy shared by all threads within a block. We can allocate such memory inside a CUDA kernel using the `__shared__` keyword.
- At the thread level, there is small amount of memory called **registers** which is thread private (not shared). It is used to store variables (scalars or small arrays of prescribed size) we declare inside our kernels and are allocated automatically. This memory cannot be accessed by the host.

### Synchronisation: histogram example

We here try to address the case where several threads attempt to write simultaneously into the same location and need to synchronise to obtain the correct output. For this we use the `atomicAdd` CUDA function. We start with a basic implementation and then try to make it faster using shared memory at the block level.

```python
import math
import numpy as np
import cupy
from cupyx.profiler import benchmark

def histogram(input_array, output_array):
  for item in input_array:
    output_array[item] = output_array[item] + 1

# input size
size = 2**25
threads_per_block = 256

# allocate memory on CPU and GPU
input_gpu = cupy.random.randint(256, size=size, dtype=cupy.int32)
input_cpu = cupy.asnumpy(input_gpu)
output_gpu = cupy.zeros(256, dtype=cupy.int32)
output_gpu_fast = cupy.zeros(256, dtype=cupy.int32)
output_cpu = cupy.asnumpy(output_gpu)

# CUDA code
histogram_cuda_code = r'''
extern "C"

__global__void histogram(const int* input, int* output){
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    atomicAdd(&(output[input[item]]), 1);
}
'''

histogram_cuda_code_fast = r'''
extern "C"

__global__void histogram(const int* input, int* output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    
    // Initialize shared memory and synchronize
    __shared__ int local_histogram[256];
    local_histogram[threadIdx.x] = 0;
    __syncthreads();
    
    // Compute shared memory histogram and synchronize
    atomicAdd(&(local_histogram[input[item]]));
    __syncthreads();

    // Update global histogram
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
'''

# compile and setup CUDA code
histogram_gpu = cupy.RawKernel(histogram_cuda_code, "histogram")
histogram_gpu_fast = cupy.RawKernel(histogram_cuda_code_fast, "histogram")
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

# check correctness
histogram(input_cpu, output_cpu)
histogram_gpu(grid_size, block_size, (input_gpu, output_gpu))
histogram_gpu_fast(grid_size, block_size, (input_gpu, output_gpu_fast))
if np.allclose(output_cpu, output_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
if np.allclose(output_cpu, output_gpu_fast):
    print("Correct results for fast implementation!")
else:
    print("Wrong results for fast implementation!")
    
# measure performance
%timeit -n 1 -r 1 histogram(input_cpu, output_cpu)
execution_gpu = benchmark(histogram_gpu, (grid_size, block_size, (input_gpu, output_gpu)), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print("Basic implementation")
print(f"{gpu_avg_time:.6f} s")
execution_gpu_fast = benchmark(histogram_gpu_fast, (grid_size, block_size, (input_gpu, output_gpu_fast)), n_repeat=10)
gpu_avg_time_fast = np.average(execution_gpu_fast.gpu_times)
print("Fast implementation exploiting shared memory")
print(f"{gpu_avg_time_fast:.6f} s")
```

## 📚 Resources

* [CodiMD Markdown Guide](https://www.markdownguide.org/tools/codimd/)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)