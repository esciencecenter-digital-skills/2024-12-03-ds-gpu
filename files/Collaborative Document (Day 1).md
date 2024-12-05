![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document (Day 1)

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
|  Time | Topic                                               |
| -----:|:--------------------------------------------------- |
|  9:30 | Welcome and icebreaker                              |
|  9:45 | Introduction                                        |
| 10:00 | Convolve an image with a kernel on a GPU using CuPy |
| 10:30 | Coffee Break                                        |
| 10:45 | Running CPU/GPU agnostic code using CuPy            |
| 11:15 | Coffee break                                        |
| 11:30 | Image processing example with CuPy                  |
| 12:00 | Lunch break                                         |
| 13:00 | Image processing example with CuPy                  |
| 14:00 | Coffee break                                        |
| 14:15 | Run your Python code on a GPU using Numba           |
| 15:00 | Coffee break                                        |
| 15:15 | Run your Python code on a GPU using Numba           |
| 16:15 | Wrap-up                                             |
| 16:30 | END                                                 |

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

### Challenge: 1d convolution

Try to run the 1-d convolution on the GPU.

#### Solution

```python
diracs_1d_gpu = cp.asarray(diracs_1d_cpu)
gauss_1d_gpu = cp.asarray(gauss_1d_cpu)

benchmark_gpu = benchmark(np.convolve, (diracs_1d_gpu, gauss_1d_gpu), n_repeat=10)
gpu_execution_avg = np.average(benchmark_gpu.gpu_times)
print(f"{gpu_execution_avg:.6f} s")
```

### Challenge: $\kappa, \sigma$ clipping on the GPU

Now that you understand how the $\kappa, \sigma$ clipping algorithm works, perform it on the GPU using CuPy and compute the speedup.


### Challenge: putting it all together

Combine the first two steps of image processing for astronomy, i.e. determining background characteristics e.g. through $\kappa, \sigma$ clipping and segmentation into a single function, that works for both CPU and GPU. Next, write a function for connected component labelling and source measurements on the GPU and calculate the overall speedup factor for the combined four steps of image processing in astronomy on the GPU relative to the CPU. Finally, verify your output by comparing with the previous output, using the CPU.

#### Our solution

```python=
def first_two_steps_for_both_CPU_and_GPU(data):
    data_flat = data.ravel()
    data_clipped = ks_clipper_cpu(data_flat)
    stddev_ = np.std(data_clipped)
    threshold = 5 * stddev_
    segmented_image = np.where(data > threshold, 1,  0)
    return segmented_image

def ccl_and_source_measurements_on_CPU(data_CPU, segmented_image_CPU):
    labeled_image_CPU = np.empty(data_CPU.shape)
    number_of_sources_in_image = label_cpu(segmented_image_CPU, 
                                           output= labeled_image_CPU)
    all_positions = com_cpu(data_CPU, labeled_image_CPU, 
                            np.arange(1, number_of_sources_in_image+1))
    all_fluxes = sl_cpu(data_CPU, labeled_image_CPU, 
                        np.arange(1, number_of_sources_in_image+1))
    return np.array(all_positions), np.array(all_fluxes)

def all_on_cpu(data):
    clipped = first_two_steps_for_both_CPU_and_GPU(data)
    return ccl_and_source_measurements_on_CPU(data, clipped)

CPU_output = all_on_cpu(data)

timing_complete_processing_CPU =  benchmark(
        all_on_cpu, (data,), n_repeat=10)

fastest_complete_processing_CPU = np.amin(timing_complete_processing_CPU.cpu_times)

print(f"The four steps of image processing for astronomy take "
      f"{1000 * fastest_complete_processing_CPU:.3e} ms\n on our CPU.")

from cupyx.scipy.ndimage import label as label_gpu
from cupyx.scipy.ndimage import center_of_mass as com_gpu
from cupyx.scipy.ndimage import sum_labels as sl_gpu

def ccl_and_source_measurements_on_GPU(data_GPU, segmented_image_GPU):
    labeled_image_GPU = cp.empty(data_GPU.shape)
    number_of_sources_in_image = label_gpu(segmented_image_GPU, 
                                           output= labeled_image_GPU)
    all_positions = com_gpu(data_GPU, labeled_image_GPU, 
                            cp.arange(1, number_of_sources_in_image+1))
    all_fluxes = sl_gpu(data_GPU, labeled_image_GPU, 
                        cp.arange(1, number_of_sources_in_image+1))
    # This seems redundant, but we want to return ndarrays (Numpy)
    # and what we have are lists.
    # These first have to be converted to
    # Cupy arrays before they can be converted to Numpy arrays.
    return cp.asnumpy(cp.asarray(all_positions)), \
           cp.asnumpy(cp.asarray(all_fluxes))


def all_on_gpu(data):
    data_gpu = cp.asarray(data)
    clipped = first_two_steps_for_both_CPU_and_GPU(data_gpu)
    return ccl_and_source_measurements_on_GPU(data_gpu, clipped)


GPU_output = all_on_gpu(data)
timing_complete_processing_GPU =  benchmark(
    all_on_gpu, (data,), n_repeat=10)
fastest_complete_processing_GPU = np.amin(
    timing_complete_processing_GPU.gpu_times)

print(f"The four steps of image processing for astronomy take "
      f"{1000 * fastest_complete_processing_GPU:.3e} ms\n on our GPU.")

overall_speedup_factor = fastest_complete_processing_CPU / fastest_complete_processing_GPU
print(f"This means that the overall speedup factor GPU vs CPU equals: {overall_speedup_factor:.3e}\n")

all_positions_agree = np.allclose(CPU_output[0], GPU_output[0])
print(f"The CPU and GPU positions agree: {all_positions_agree}\n")

all_fluxes_agree = np.allclose(CPU_output[1], GPU_output[1])
print(f"The CPU and GPU fluxes agree: {all_positions_agree}\n")
```

### Challenge: compute prime numbers

Write a new function `find_all_primes_cpu_and_gpu` that uses `check_prime_gpu_kernel` instead of the inner loop of `find_all_primes_cpu`. How long does this new function take to find all primes up to 10000?


#### Our solution
```python
def find_all_primes_gpu(upper):
    all_prime_numbers = []
    for num in range(0, upper):
        result = np.zeros(1, np.int32)
        check_prime_gpu_kernel[1,1](num, result)
        if result[0] != 0:
            all_prime_numbers.append(num)
    return all_prime_numbers

%timeit -n 10 -r 1 find_all_primes_gpu(10_000)
```
Let's consider an alternative and faster solution using numba vectorize so that we can work on arrays and benefit from the GPU.
```python
import numba as nb

@nb.vectorize(['int32(int32)'], target='cuda')
def check_prime_gpu(num):
    if num == 1:
       return 0
    for i in range(2, (num // 2) + 1):
       if (num % i) == 0:
           return 0
    return num

numbers = np.arange(0, 10_000, dtype=np.int32)
list(filter(lambda x: x !=0, check_prime_gpu(numbers)))
%timeit -n 10 -r 1 check_prime_gpu(numbers)
```
Alternative formulation where we explicitly assign  numbers to GPU threads
```python
@cuda.jit
def check_prime_gpu_kernel(result):
    num = cuda.grid(1)
    result[num] = num
    if num == 1:
        result[num] = 0
    for i in range(2, (num // 2) + 1):
       if (num % i) == 0:
           result[num] = 0
           return 0

result = np.zeros(10000, dtype=np.int32)
check_prime_gpu_kernel[100,100][result]
```

## 🧠 Collaborative Notes

Check the GPU:

```
!nvidia-smi
```

We install dependencies:

```
!pip install cupy-cuda12x
!pip install numba astropy scipy matplotlib
```

### Convoluted Example

We'll start with an image that combines some dirac-delta functions.

```python
import numpy as np

diracs = np.zeros([2048, 2048])
diracs[8:16,8:16] = 1
```

Now visualize

```python
import pylab as pyl
%matplotlib inline
pyl.imshow(diracs[0:32,0:32])
```

We will convolve this image with a Gaussian kernel now. First we prepare the kernel.

```python
x, y = np.meshgrid(np.linspace(-2, 2, 15), np.linspace(-2, 2, 15))
dist = np.sqrt(x*x + y*y)
sigma = 1
origin = 0.0
gauss = np.exp(-(dist - origin)**2 / (2.0 * sigma**2))
pyl.imshow(gauss)
pyl.show()
```

First, we'll do this on the CPU using the SciPy library.

```python
from scipy.signal import convolve2d as convolve2d_cpu

convolved_image_cpu = convolve2d_cpu(diracs, gauss)

# add 7 to the indices to skip the margin
pyl.imshow(convolved_image_cpu[7:39, 7:39])
pyl.show()
```

Timing the result:

```python
%timeit -n 1 -r 1 convolve2d_cpu(diracs, gauss)
```

Now on the GPU: we first need to copy the data to the GPU.

```python
import cupy as cp

diracs_gpu = cp.asarray(diracs)
gauss_gpu = cp.asarray(gauss)
```

The CuPyX package has a nice function for convolutions.

```python
from cupyx.scipy.signal import convolve2d as convolve2d_gpu

convolved_image_gpu = convolve2d_gpu(diracs_gpu, gauss_gpu)
```

Let's check that this worked. We can obtain the data from the GPU using the `.get()` method.

```python
pyl.imshow(convolved_image_gpu[0:64, 0:64].get())
```

Visuals is nice and all, but are the values the same?

```python
np.allclose(convolved_image_cpu, convolved_image_gpu)
```

Is this any faster?

```python
%timeit -n 1 -r 5 convolve2d_gpu(diracs_gpu, gauss_gpu)
```

Hmmm, `%timeit` seems to be broken here. The CPU isn't waiting for the GPU to be done. 

```python
from cupyx.profiler import benchmark

benchmark_gpu = benchmark(convolve2d_gpu, (diracs_gpu, gauss_gpu), n_repeat=10)
gpu_execution_times = np.average(benchmark_gpu.gpu_times)
print(f"{gpu_execution_times:.6f} s")
```

### Using NumPy directly

```python
diracs_1d_cpu = diracs.ravel()
gauss_1d_cpu = gauss.diagonal()
np.convolve(diracs_1d_cpu, gauss_1d_cpu)
```

### Timing with data transfer

```python
def gpu_convolution_with_data_transfer():
    diracs_gpu = cp.asarray(diracs)
    gauss_gpu = cp.asarray(gauss)
    convolved_image_gpu = convolve2d_gpu(diracs_gpu, gauss_gpu)
    convolved_image_in_host = cp.asnumpy(convolved_image_gpu)
    
benchmark_gpu = benchmark(gpu_convolution_with_data_transfer, (), n_repeat=10)
gpu_execution_avg = np.average(benchmark_gpu.gpu_times)
print(f"{gpu_execution_avg:.6f} s")
```

For this specific benchmark, we can actually use `%timeit`, since everything is synchronized before and after the computation.

### Application: Astronomy

Download this image: https://carpentries-incubator.github.io/lesson-gpu-programming/data/GMRT_image_of_Galactic_Center.fits

```
!wget https://carpentries-incubator.github.io/lesson-gpu-programming/data/GMRT_image_of_Galactic_Center.fits
```

Load the image.

```python
from astropy.io import fits

with fits.open("GMRT_image_of_Galactic_Center.fits") as hdul:
    data = hdul[0].data.byteswap().newbyteorder()
```

Show the image.

```python
from matplotlib.colors import LogNorm

maxim = data.max()

fig = pyl.figure(figsize=(50, 12.5))
ax = fig.add_subplot(1, 1, 1)
im_plot = ax.imshow(np.fliplr(data), cmap=pyl.cm.gray_r, norm=LogNorm(vmin = maxim/10, vmax=maxim/100))
pyl.colorbar(im_plot, ax=ax)
```

Get some summary stats.

```python
mean_ = data.mean()
median_ = np.median(data)
stddev_ = np.std(data)
max_ = np.amax(data)
print(f"mean = {mean_:.3e}, median = {median_:.3e}, sttdev = {stddev_:.3e}, maximum = {max_:.3e}")
```

#### $\kappa-\sigma$ clipping

```python
# Flattening our 2D data makes subsequent steps easier
data_flat = data.ravel()

# Here is a kappa-sigma clipper for the CPU
def ks_clipper_cpu(data_flat):
    while True:
         med = np.median(data_flat)
         std = np.std(data_flat)
         clipped_below = data_flat.compress(data_flat > med - 3 * std)
         clipped_data = clipped_below.compress(clipped_below < med + 3 * std)
         if len(clipped_data) == len(data_flat):
             break
         data_flat = clipped_data  
    return data_flat

data_clipped_cpu = ks_clipper_cpu(data_flat)
timing_ks_clipping_cpu = %timeit -o ks_clipper_cpu(data_flat)
fastest_ks_clipping_cpu = timing_ks_clipping_cpu.best
print(f"Fastest ks clipping time on CPU = {1000 * fastest_ks_clipping_cpu:.3e} ms.")
```

Statistics after clipping:

```python
clipped_mean_ = data_clipped_cpu.mean()
clipped_median_ = np.median(data_clipped_cpu)
clipped_stddev_ = np.std(data_clipped_cpu)
clipped_max_ = np.amax(data_clipped_cpu)
print(f"mean of clipped = {clipped_mean_:.3e}, "
      f"median of clipped = {clipped_median_:.3e} \n"
      f"standard deviation of clipped = {clipped_stddev_:.3e},"
      f"maximum of clipped = {clipped_max_:.3e}")
```

#### Step 2: Image segmentation

```python
threshold = 5 * clipped_stddev_
segmented_image = np.where(data > threshold, 1,  0)
```

#### Step 3: Labeling sources

```python
from scipy.ndimage import label as label_cpu
labeled_image = np.empty(data.shape)
number_of_sources_cpu = label_cpu(segmented_image, output=labeled_image)
sigma_unicode = "\u03C3"
print(f"The number of sources in the image at the 5{sigma_unicode} level is {number_of_sources_cpu}")
```

#### Step 4: Measuring radiation from sources

```python
from scipy.ndimage import center_of_mass as com_cpu
from scipy.ndimage import sum_labels as sl_cpu
all_positions = com_cpu(data, labeled_image,
                        range(1, number_of_sources_cpu+1))
all_integrated_fluxes = sl_cpu(data, labeled_image,
                               range(1, number_of_sources_cpu+1))

print (f'These are the ten highest integrated fluxes of the sources in my \n image: {np.sort(all_integrated_fluxes)[-10:]}')
```

### Using Numba to execute Python code on the GPU
We start with a function which gets all prime numbers in a range of integers.
```python
def find_all_primes_cpu(upper):
    all_prime_numbers = []
    for num in range(0, upper):
        prime = True
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                prime = False
                break
        if prime:
            all_prime_numbers.append(num)
    return all_prime_numbers
```
Let's see how much time it takes to find all prime numbers up to 10,000.
```python
%timeit -n 10 -r 1 find_all_primes_cpu(10_000)
```
We now use the just in time (jit) numba tool to speed up the execution of this function
```python
from numba import jit
find_all_prime_numba = jit(find_all_primes_cpu)
%timeit -n 10 -r 1 find_all_primes_numba(10_000)
```
Numba also provides a way to convert and use your code on the GPU. We write here a kernel that checks if a number is a prime number or not
```python
from numba import cuda
@cuda.jit
def check_prime_gpu_kernel(num, result):
   result[0] =  num
   for i in range(2, (num // 2) + 1):
       if (num % i) == 0:
           result[0] = 0
           break
```
We had to adapt our function because CUDA kernels do not return anything. Let's use our GPU function to check whether 11 and 12 are prime numbers:
```python
import numpy as np

result = np.zeros((1), np.int32)
check_prime_gpu_kernel[1, 1](11, result)
print(result[0])
check_prime_gpu_kernel[1, 1](12, result)
print(result[0])
```

### How does the GPU hardware work
Watch the video:
https://www.youtube.com/watch?v=FcS_kQOIykU

## 📚 Resources

* [HackMD Markdown Guide](https://www.markdownguide.org/tools/hackmd/)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [Numba User Guide](https://numba.pydata.org/numba-doc/latest/user/index.html)
* [CuPy documentation](https://cupy.dev/)