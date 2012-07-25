# Author: Martin Smith
# Created on: 7/18/12
# Updated on: 7/18/12

import pyopencl as cl
import pyopencl.array as cla
import numpy as np
import sys


def round_up(number, multiple=1):
    if number == round_down(number, multiple):
        return number
    else:
        return round_down(number+multiple, multiple)


def round_down(number, multiple=1):
    return int(number/multiple)*multiple


def round_unitI_test():
    print("Round Down: \n\t(exp,res): (64,%i)" % round_down(72,64))
    print("Round Down: \n\t(exp,res): (0,%i)" % round_down(64,72))
    print("Round Down: \n\t(exp,res): (64,%i)" % round_down(64,64))

    print("Round Up: \n\t(exp,res): (128,%i)" % round_up(72,64))
    print("Round Up: \n\t(exp,res): (72,%i)" % round_up(64,72))
    print("Round Up: \n\t(exp,res): (64,%i)" % round_up(64,64))


def pad(matrix, multiple=64):
    bup_shape = matrix.shape
    for dim in range(2):
        if round_up(matrix.shape[dim], multiple) != matrix.shape[dim]:
            #print("Adjustments: ")
            #print(matrix.shape)
            appendShape = (round_up(matrix.shape[dim], multiple)-matrix.shape[dim], matrix.shape[1-dim])
            if dim:
                appendShape = (appendShape[1], appendShape[0])
            toAppend = np.zeros(appendShape)
            #print(toAppend.shape)
            matrix = np.append(matrix, toAppend, dim)
            #print(matrix.shape)
    return matrix, bup_shape


def matrix_multiply(A,B):
    ###################################################
    #### CL setup

    devices = get_devices()
    preferred = 'GPU'
    try:
        dev = devices[preferred]
        dev_type = preferred
        print("Using: ", dev)
        if dev_type == 'GPU':
            C = use_GPU(dev, A, B)
        else:
            C = use_CPU(dev, A, B)
    except KeyError:
        dev,dev_type = devices.getitems()[0]
        print("Using: ", dev)
        if dev_type == 'GPU':
            C = use_GPU(dev, A, B)
        else:
            C = use_CPU(dev, A, B)
    return C



def get_devices():
    if len(cl.get_platforms()) > 1:
        for found_platform in cl.get_platforms():
            if found_platform.name == 'NVIDIA CUDA':
                my_platform = found_platform
                print("Selected platform:", my_platform.name)
    else: my_platform = cl.get_platforms()[0]

    devices = {}
    for device in my_platform.get_devices():
      devices[cl.device_type.to_string(device.type)] = device
    return devices


def use_CPU(dev, A, B):
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    #return use_naive_mapped_kernel(ctx, queue, dev, A, B)
    return use_naive_kernel(ctx, queue, dev, A, B)


def use_GPU(dev, A, B):
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    #return use_linear_opt_kernel(ctx, queue, dev, A, B) # not working
    #return use_naive_kernel(ctx, queue, dev, A, B)
    #return use_twoD_opt_kernel(ctx, queue, dev, A, B)
    return use_speed_junk_kernel(ctx, queue, dev, A, B)
    #return use_dual_twoD_opt_kernel(ctx, queue, dev, A, B)


def use_linear_opt_kernel(ctx, queue, dev, A, B):
    newA, A_shape = pad(A.copy())
    newB, B_shape = pad(B.copy())

    C_shape = (A.shape[0], B.shape[1])
    newC_shape = (newA.shape[0], newB.shape[1])
    newC = np.zeros(newC_shape, dtype=np.float32)

    A_cache = np.array(newA.flatten(), dtype=np.float32)
    B_cache = np.array(newB.flatten(), dtype=np.float32)
    C_cache = np.array(newC.flatten(), dtype=np.float32)

    A_array = cla.to_device(queue, A_cache)
    B_array = cla.to_device(queue, B_cache)
    C_array = cla.to_device(queue, C_cache)

    max_wg_size = dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

    global_size = (round_up(C_array.shape[0], max_wg_size),)
    blocksize = 32
    local_size = (blocksize * blocksize,)

    print("Local Size: ", local_size)
    print("Global Size: ", global_size)

    kernel = linear_opt_kernel()
    prg = cl.Program(ctx, kernel).build()
    event =  prg.matMul( queue,
                        global_size,
                        local_size,
                        A_array.data,
                        B_array.data,
                        C_array.data,
                        np.int32(A_shape[1]),
                        np.int32(newA.shape[1]),
                        np.int32(newC.shape[1]),
                        np.int32(C_shape[0]), # row boundary
                        np.int32(C_shape[1])) # col boundary
    event.wait()
    C_cache = C_array.get().reshape(newC_shape)
    return C_cache[: C_shape[0], : C_shape[1]]


def linear_opt_kernel():
    return """
#define BLOCKSIZE 32
__kernel void matMul(
                __global float* A,
                __global float* B,
                __global float* C,
                int a_row_len,
                int a_round_row_len,
                int c_round_row_len,
                int row_bound,
                int col_bound)
{
    // Block-wise caching of the source matrices.
    int g_id = get_global_id(0);
    int g_col = g_id % c_round_row_len; // could these modifications be reducing performance?
    int g_row = g_id / c_round_row_len;
    int block_start_row = g_row / BLOCKSIZE;
    int block_start_col = g_col / BLOCKSIZE;
    int l_id = get_local_id(0);
    int l_col = l_id % BLOCKSIZE;
    int l_row = l_id / BLOCKSIZE;
    __local float A_cache[BLOCKSIZE*BLOCKSIZE];
    __local float B_cache[BLOCKSIZE*BLOCKSIZE];

    float sum = 0.0;
    for (int step = 0; step < (int)(a_round_row_len/BLOCKSIZE); step++)
    {
        A_cache[l_id] = A[g_row * a_round_row_len + step * BLOCKSIZE + l_col];
        B_cache[l_id] = B[g_col + step * c_round_row_len * BLOCKSIZE + l_row];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < BLOCKSIZE; i++)
            sum += A_cache[(l_row * BLOCKSIZE) + i] * B_cache[(i * BLOCKSIZE) + l_col];
    }
    if (g_row < row_bound && g_col < col_bound)
        C[g_id] = l_row;
}
    """


def use_dual_twoD_opt_kernel(ctx, queue, dev, A, B):
    newA, A_shape = pad(A.copy())
    newB, B_shape = pad(B.copy())

    C_shape = (A.shape[0], B.shape[1])
    newC_shape = (newA.shape[0], newB.shape[1])
    newC = np.zeros(newC_shape, dtype=np.float32)

    A_cache = np.array(newA.flatten(), dtype=np.float32)
    B_cache = np.array(newB.flatten(), dtype=np.float32)
    C_cache = np.array(newC.flatten(), dtype=np.float32)

    A_array = cla.to_device(queue, A_cache)
    B_array = cla.to_device(queue, B_cache)
    C_array = cla.to_device(queue, C_cache)

    max_wg_size = dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

    global_size = (newC_shape[0], newC_shape[1])
    blocksize = 32
    local_size = (blocksize, blocksize)

    print("Local Size: ", local_size)
    print("Global Size: ", global_size)

    kernel = dual_twoD_opt_kernel()
    prg = cl.Program(ctx, kernel).build()
    event =  prg.matMul( queue,
                        global_size,
                        local_size,
                        A_array.data,
                        B_array.data,
                        C_array.data,
                        np.int32(A_shape[1]),
                        np.int32(newA.shape[1]),
                        np.int32(newC.shape[1]),
                        np.int32(C_shape[0]), # row boundary
                        np.int32(C_shape[1])) # col boundary
    event.wait()
    C_cache = C_array.get().reshape(newC_shape)
    return C_cache[: C_shape[0], : C_shape[1]]


def dual_twoD_opt_kernel():
    return """
#define BLOCKSIZE 32
__kernel void matMul(
                __global float* A,
                __global float* B,
                __global float* C,
                int a_row_len,
                int a_round_row_len,
                int c_round_row_len,
                int row_bound,
                int col_bound)
{
    int g_col = get_global_id(0);
    int g_row = get_global_id(1);
    int l_col = get_local_id(0);
    int l_row = get_local_id(1);
    int block_start_row = g_row / BLOCKSIZE;
    int block_start_col = g_col / BLOCKSIZE;

    __local float A_cache[BLOCKSIZE*BLOCKSIZE];
    __local float B_cache[BLOCKSIZE*BLOCKSIZE];

    float sum = 0.0;
    for (int step = 0; step < a_round_row_len/BLOCKSIZE; step++)
    {
        A_cache[get_local_id(0)] = A[block_start_row * a_round_row_len * BLOCKSIZE + step * BLOCKSIZE + l_row * a_round_row_len + l_col];
        B_cache[get_local_id(0)] = B[block_start_col * BLOCKSIZE + step * c_round_row_len * BLOCKSIZE + l_row * c_round_row_len + l_col];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < BLOCKSIZE; i++)
            sum += A_cache[(l_row * BLOCKSIZE) + i] * B_cache[(i * BLOCKSIZE) + l_col];
    }
    if (g_row < row_bound && g_col < col_bound)
        C[g_row * c_round_row_len + g_col] = sum;
}
    """

def use_speed_junk_kernel(ctx, queue, dev, A, B):
    newA, A_shape = pad(A.copy())
    newB, B_shape = pad(B.copy())

    C_shape = (A.shape[0], B.shape[1])
    newC_shape = (newA.shape[0], newB.shape[1])
    newC = np.zeros(newC_shape, dtype=np.float32)

    A_cache = np.array(newA.flatten(), dtype=np.float32)
    B_cache = np.array(newB.flatten(), dtype=np.float32)
    C_cache = np.array(newC.flatten(), dtype=np.float32)

    A_array = cla.to_device(queue, A_cache)
    B_array = cla.to_device(queue, B_cache)
    C_array = cla.to_device(queue, C_cache)

    max_wg_size = dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

    global_size = (newC_shape[0], newC_shape[1])
    blocksize = 16
    local_size = (blocksize, blocksize)

    print("Local Size: ", local_size)
    print("Global Size: ", global_size)

    kernel = speed_junk_kernel()
    prg = cl.Program(ctx, kernel).build()
    event =  prg.matMul( queue,
                        global_size,
                        local_size,
                        A_array.data,
                        B_array.data,
                        C_array.data,
                        np.int32(A_shape[1]),
                        np.int32(newA.shape[1]),
                        np.int32(newC.shape[1]),
                        np.int32(C_shape[0]), # row boundary
                        np.int32(C_shape[1])) # col boundary
    event.wait()
    C_cache = C_array.get().reshape(newC_shape)
    return C_cache[: C_shape[0], : C_shape[1]]


def speed_junk_kernel():
    return """
#define BLOCK_SIZE 16
__kernel void matMul(
                __global float* A,
                __global float* B,
                __global float* C,
                int a_row_len,
                int a_round_row_len,
                int c_round_row_len,
                int row_bound,
                int col_bound)
{
    int wA = a_round_row_len;
    int wB = c_round_row_len;

    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    // Index of the first sub-matrix of A processed
    // by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed
    // by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the
    // sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed
    // by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the
    // sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0.0f;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep)
    {

        // Declaration of the local memory array As
        // used to store the sub-matrix of A
        __local float As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the local memory array Bs
        // used to store the sub-matrix of B
        __local float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from global memory
        // to local memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices
        // are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[ty][k] * Bs[k][tx];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);

    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;

}
    """


def use_twoD_opt_kernel(ctx, queue, dev, A, B):
    newA, A_shape = pad(A.copy())
    newB, B_shape = pad(B.copy())

    C_shape = (A.shape[0], B.shape[1])
    newC_shape = (newA.shape[0], newB.shape[1])
    newC = np.zeros(newC_shape, dtype=np.float32)

    A_cache = np.array(newA.flatten(), dtype=np.float32)
    B_cache = np.array(newB.flatten(), dtype=np.float32)
    C_cache = np.array(newC.flatten(), dtype=np.float32)

    A_array = cla.to_device(queue, A_cache)
    B_array = cla.to_device(queue, B_cache)
    C_array = cla.to_device(queue, C_cache)

    max_wg_size = dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

    global_size = (newC_shape[0], newC_shape[1])
    blocksize = 16
    local_size = (blocksize, blocksize)

    print("Local Size: ", local_size)
    print("Global Size: ", global_size)

    kernel = twoD_opt_kernel()
    prg = cl.Program(ctx, kernel).build()
    event =  prg.matMul( queue,
                        global_size,
                        local_size,
                        A_array.data,
                        B_array.data,
                        C_array.data,
                        np.int32(A_shape[1]),
                        np.int32(newA.shape[1]),
                        np.int32(newC.shape[1]),
                        np.int32(C_shape[0]), # row boundary
                        np.int32(C_shape[1])) # col boundary
    event.wait()
    C_cache = C_array.get().reshape(newC_shape)
    return C_cache[: C_shape[0], : C_shape[1]]


def twoD_opt_kernel():
    return """
#define BLOCKSIZE 16
#define BLOCK_SIZE 16
#define AS(i, j) As[j + i * BLOCK_SIZE]
#define BS(i, j) Bs[j + i * BLOCK_SIZE]
__kernel void matMul(
                __global float* A,
                __global float* B,
                __global float* C,
                int a_row_len,
                int a_round_row_len,
                int c_round_row_len,
                int row_bound,
                int col_bound)
{
    // Block index
    int bx = get_group_id(0);
    int by = get_group_id(1);

    // Thread index
    int tx = get_local_id(0);
    int ty = get_local_id(1);

    __local float As[BLOCK_SIZE * BLOCK_SIZE];
    __local float Bs[BLOCK_SIZE * BLOCK_SIZE];

    int uiWA = a_round_row_len;
    int uiWB = c_round_row_len;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = uiWA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + uiWA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * uiWB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0.0f;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A[a + uiWA * ty + tx];
        BS(ty, tx) = B[b + uiWB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += AS(ty, k) * BS(k, tx);

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    int trueLocalSize1 = col_bound;

    if (get_global_id(1) < trueLocalSize1)
    // Write the block sub-matrix to device memory;
    // each thread writes one element
    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub;
/*
    int g_col = get_global_id(0);
    int g_row = get_global_id(1);
    int l_col = get_local_id(0);
    int l_row = get_local_id(1);
    int block_num_X = get_group_id(0);
    int block_num_Y = get_group_id(1);

    __local float A_cache[BLOCKSIZE*BLOCKSIZE];
    __local float B_cache[BLOCKSIZE*BLOCKSIZE];

    float sum = 0.0;
    for (int step = 0; step < a_round_row_len/BLOCKSIZE; step++)
    {
        A_cache[g_row * BLOCKSIZE + g_col] = A[block_num_Y * a_round_row_len * BLOCKSIZE + step * BLOCKSIZE + l_row * a_round_row_len + l_col];
        B_cache[g_row * BLOCKSIZE + g_col] = B[block_num_X * BLOCKSIZE + step * c_round_row_len * BLOCKSIZE + l_row * c_round_row_len + l_col];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < BLOCKSIZE; i++)
            sum += A_cache[(l_row * BLOCKSIZE) + i] * B_cache[(i * BLOCKSIZE) + l_col];
    }
    if (g_row < row_bound && g_col < col_bound)
        C[g_row * c_round_row_len + g_col] = sum;
*/
}
    """

def use_naive_mapped_kernel(ctx, queue, dev, A, B):
    newA, A_shape = pad(A.copy())
    newB, B_shape = pad(B.copy())

    C_shape = (A.shape[0], B.shape[1])
    newC_shape = (newA.shape[0], newB.shape[1])
    newC = np.zeros(newC_shape, dtype=np.float32)

    A_cache = np.array(newA.flatten(), dtype=np.float32)
    B_cache = np.array(newB.flatten(), dtype=np.float32)
    C_cache = np.array(newC.flatten(), dtype=np.float32)

    max_wg_size = dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

    A_array = cla.to_device(queue, A_cache)
    B_array = cla.to_device(queue, B_cache)
    C_array = cla.to_device(queue, C_cache)

    global_size = (round_up(C_cache.shape[0], max_wg_size),)
    local_size = None

    print("Local Size: ", local_size)
    print("Global Size: ", global_size)

    kernel = naive_mapped_kernel()
    prg = cl.Program(ctx, kernel).build()

    event = prg.matMul( queue,
                        global_size,
                        local_size,
                        A_array.data,
                        B_array.data,
                        C_array.data,
                        np.int32(A_shape[1]),
                        np.int32(newC.shape[1]),
                        np.int32(C_shape[0]), # row boundary
                        np.int32(C_shape[1])) # col boundary
    event.wait()
    C_cache = C_array.get().reshape(newC_shape)
    return C_cache[: C_shape[0], : C_shape[1]]


def naive_mapped_kernel():
    return """

__kernel void matMul(
                __global float* A,
                __global float* B,
                __global float* C,
                int aRowLen,
                int cRowLen,
                int rowBound,
                int colBound)
{
    int gid = get_global_id(0);
    int col = gid % cRowLen;
    int row = gid / cRowLen;

    float sum = 0.0;
    for (int i = 0; i < aRowLen; i++)
    {
        sum += A[(row * cRowLen) + i] * B[(i * cRowLen) + col];
    }
    if (row < rowBound && col < colBound)
        C[gid] = sum;
}
    """


def use_naive_kernel(ctx, queue, dev, A, B):
    newA, A_shape = pad(A.copy())
    newB, B_shape = pad(B.copy())

    C_shape = (A.shape[0], B.shape[1])
    newC_shape = (newA.shape[0], newB.shape[1])
    newC = np.zeros(newC_shape, dtype=np.float32)

    A_cache = np.array(newA.flatten(), dtype=np.float32)
    B_cache = np.array(newB.flatten(), dtype=np.float32)
    C_cache = np.array(newC.flatten(), dtype=np.float32)

    max_wg_size = dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    kernel = naive_kernel()

    mf = cl.mem_flags
    flags = mf.READ_WRITE | mf.COPY_HOST_PTR | mf.ALLOC_HOST_PTR
    A_buffer = cl.Buffer(ctx, flags, hostbuf=A_cache)
    B_buffer = cl.Buffer(ctx, flags, hostbuf=B_cache)
    C_buffer = cl.Buffer(ctx, flags, hostbuf=C_cache)
    A_array, _ = cl.enqueue_map_buffer(queue, A_buffer, cl.map_flags.READ, 0, A_cache.shape, A_cache.dtype, "C")
    B_array, _ = cl.enqueue_map_buffer(queue, B_buffer, cl.map_flags.READ, 0, B_cache.shape, B_cache.dtype, "C")
    C_array, _ = cl.enqueue_map_buffer(queue, C_buffer, cl.map_flags.WRITE, 0, C_cache.shape, C_cache.dtype, "C")

    global_size = (round_up(C_cache.shape[0], max_wg_size),)
    local_size = None

    print("Local Size: ", local_size)
    print("Global Size: ", global_size)

    prg = cl.Program(ctx, kernel).build()

    event = prg.naiveMatMul( queue,
                        global_size,
                        local_size,
                        A_array.data,
                        B_array.data,
                        C_array.data,
                        np.int32(A_shape[1]),
                        np.int32(newC.shape[1]),
                        np.int32(C_shape[0]), # row boundary
                        np.int32(C_shape[1])) # col boundary
    event.wait()
    cl.enqueue_copy(queue, C_cache, C_array)
    return C_cache.reshape(newC_shape)[: C_shape[0], : C_shape[1]]


def naive_kernel():
    return """

__kernel void naiveMatMul(
                __global float* A,
                __global float* B,
                __global float* C,
                int aRowLen,
                int cRowLen,
                int rowBound,
                int colBound)
{
    int gid = get_global_id(0);
    int col = gid % cRowLen;
    int row = gid / cRowLen;

    float sum = 0.0;
    for (int i = 0; i < aRowLen; i++)
    {
        sum += A[(row * cRowLen) + i] * B[(i * cRowLen) + col];
    }
    if (row < rowBound && col < colBound)
        C[gid] = sum;
}
    """
