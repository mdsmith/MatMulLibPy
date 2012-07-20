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
    # Platform test
    deviceOfChoice = 'GPU'
    #deviceOfChoice = 'CPU'
    if len(sys.argv) > 1:
        deviceOfChoice = sys.argv[1]
    print("Platform of choice: ", deviceOfChoice)

    if len(cl.get_platforms()) > 1:
        for found_platform in cl.get_platforms():
            if found_platform.name == 'NVIDIA CUDA':
                my_platform = found_platform
                print("Selected platform:", my_platform.name)
    else: my_platform = cl.get_platforms()[0]

    for device in my_platform.get_devices():
      dev_type = cl.device_type.to_string(device.type)
      if dev_type == deviceOfChoice:
            dev = device
            print("Selected device: ", dev_type, device)

    max_wg_size = dev.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    ctx = cl.Context([dev])
    queue = cl.CommandQueue(ctx)
    kernel = kernel_text()

    ####
    ###################################################
    #### Array setup

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

    #for shape_name, shape in {'A_shape':A_shape, 'B_shape':B_shape,
       #'C_shape':C_shape, 'newA_shape':newA.shape, 'newB_shape':newB.shape,
       #'newC_shape':newC.shape}.items():
       #print("%s: " % shape_name, shape)

    ####
    ###################################################

    global_size = (round_up(C_cache.shape[0], max_wg_size),)
    if deviceOfChoice == 'CPU':
        local_size = None
        #local_size = (16,)
    else:
        local_size = (min(max_wg_size, global_size[0]),)

    print("Local Size: ", local_size)
    print("Global Size: ", global_size)

    # XXX fix algorithm to not use the global x and y as location identifiers.
    # XXX fix algorithm to kill useless threads

    prg = cl.Program(ctx, kernel).build()

    event = prg.MatMul( queue,
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

    #print(newA)
    #print(newB)
    C_cache = C_array.get().reshape(newC_shape)
    #print(C_cache)
    return C_cache[: C_shape[0], : C_shape[1]]



def kernel_text():
    return """

__kernel void MatMul(
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
