from modal import App, Image, gpu, Volume
import subprocess
import textwrap
import time

app_name = "ptx-microbench"
app = App(app_name)

# --- CUDA source (compiled inside container) ---
CUDA_SRC_ADD = r"""
extern "C" __global__
void bench_add(float *out, int iters) {
    float inc = 1.0f;
    float x = threadIdx.x;

    float y = threadIdx.x+1;
    float z = threadIdx.x+2;
    float w = threadIdx.x+3;

    float v = threadIdx.x+4;
    float u = threadIdx.x+5;
    float s = threadIdx.x+6;
    float t = threadIdx.x+7;
    for (int i = 0; i < iters; i++) {
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(x) : "f"(inc));

        asm volatile ("add.f32 %0, %0, %1;" : "+f"(y) : "f"(inc));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(z) : "f"(inc));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(w) : "f"(inc));

        asm volatile ("add.f32 %0, %0, %1;" : "+f"(v) : "f"(inc));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(u) : "f"(inc));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(s) : "f"(inc));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(t) : "f"(inc));
    }
    out[threadIdx.x] = x;

    out[threadIdx.x+128] = y;
    out[threadIdx.x+2*128] = z;
    out[threadIdx.x+3*128] = w;

    out[threadIdx.x+4*128] = v;
    out[threadIdx.x+5*128] = u;
    out[threadIdx.x+6*128] = s;
    out[threadIdx.x+7*128] = t;
}
"""
CUDA_SRC_ADD2 = r"""
extern "C" __global__
void bench_add(float *out, float *in1, float* in2, int iters) {
    float x = in1[threadIdx.x];
    float y = in2[threadIdx.x];
    float z = in1[(threadIdx.x+1) % blockDim.x];
    float w = in2[(threadIdx.x+1) % blockDim.x];
    float o1 = 0.0;
    float o2 = 0.0;
    for (int i = 0; i < iters; i++) {
        asm volatile ("add.f32 %0, %1, %2;" : "+f"(o1) : "f"(x), "f"(y));
        asm volatile ("add.f32 %0, %1, %2;" : "+f"(o2) : "f"(z), "f"(w));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(x) : "f"(o1));
        asm volatile ("add.f32 %0, %0, %1;" : "+f"(y) : "f"(o2));
    }
    out[threadIdx.x] = x + y;
}
"""

CUDA_SRC_MUL = r"""
extern "C" __global__
void bench_mul(float *out, int iters) {
    float x = threadIdx.x;
    for (int i = 0; i < iters; i++) {
        asm volatile ("mul.f32 %0, %0, 1.0;" : "+f"(x));
    }
    out[threadIdx.x] = x;
}
"""

CUDA_SRC_EX2 = r"""
extern "C" __global__
void bench_ex2(float *out, int iters) {
    float x = threadIdx.x;
    for (int i = 0; i < iters; i++) {
        asm volatile ("ex2.approx.f32 %0, %0;" : "+f"(x));
    }
    out[threadIdx.x] = x;
}
"""

CUDA_SRC_EX22 = r"""
extern "C" __global__
void bench_ex2(float *out, int iters) {
    float x = 1.0f + (float)threadIdx.x * 0.00001f; // Slight variation
    float y = 1.0f + (float)threadIdx.x * 0.00002f; // Slight variation
    float z = 1.0f + (float)threadIdx.x * 0.00003f; // Slight variation
    float w = 1.0f + (float)threadIdx.x * 0.00004f; // Slight variation
    float u = 1.0f + (float)threadIdx.x * 0.00005f; // Slight variation
    float v = 1.0f + (float)threadIdx.x * 0.00006f; // Slight variation
    float s = 1.0f + (float)threadIdx.x * 0.00007f; // Slight variation
    float t = 1.0f + (float)threadIdx.x * 0.00008f; // Slight variation

    for (int i = 0; i < iters; i++) {
        // Chain: x = 2^(x - 1.0)
        asm volatile ("add.f32 %0, %0, -1.0;" :  "+f"(x));
        asm volatile ("ex2.approx.f32 %0, %0;" : "+f"(x));
        asm volatile ("add.f32 %0, %0, -1.0;" :  "+f"(y));
        asm volatile ("ex2.approx.f32 %0, %0;" : "+f"(y));
        asm volatile ("add.f32 %0, %0, -1.0;" :  "+f"(z));
        asm volatile ("ex2.approx.f32 %0, %0;" : "+f"(z));
        asm volatile ("add.f32 %0, %0, -1.0;" :  "+f"(w));
        asm volatile ("ex2.approx.f32 %0, %0;" : "+f"(w));
        asm volatile ("add.f32 %0, %0, -1.0;" :  "+f"(u));
        asm volatile ("ex2.approx.f32 %0, %0;" : "+f"(u));
        asm volatile ("add.f32 %0, %0, -1.0;" :  "+f"(v));
        asm volatile ("ex2.approx.f32 %0, %0;" : "+f"(v));
        asm volatile ("add.f32 %0, %0, -1.0;" :  "+f"(s));
        asm volatile ("ex2.approx.f32 %0, %0;" : "+f"(s));
        asm volatile ("add.f32 %0, %0, -1.0;" :  "+f"(t));
        asm volatile ("ex2.approx.f32 %0, %0;" : "+f"(t));
    }
    out[threadIdx.x] = x;
    out[threadIdx.x+128] = y;
    out[threadIdx.x+128*2] = z;
    out[threadIdx.x+128*3] = w;
    out[threadIdx.x+128*4] = u;
    out[threadIdx.x+128*5] = v;
    out[threadIdx.x+128*6] = s;
    out[threadIdx.x+128*7] = t;
}
"""

CUDA_SRC_FMA = r"""
extern "C" __global__
void bench_fma(float *out, int iters) {
    float x = threadIdx.x * 1.0f;  // initialize differently per thread
    float y = 1.0001f;
    float z = 0.9999f;

    for (int i = 0; i < iters; i++) {
        // fused multiply-add: x = x * y + z
        asm volatile("fma.rn.f32 %0, %1, %2, %0;"
                     : "+f"(x)
                     : "f"(y), "f"(z));
    }

    out[threadIdx.x] = x;
}
"""

CUDA_SRC_CVT = r"""
extern "C" __global__
void bench_cvt(float *out, int iters) {
    // Initialize a single-precision float per thread.
    // The value is chosen to ensure it exercises the conversion logic.
    float x = threadIdx.x * 0.0001f + 1.23456f;
    float y = threadIdx.x * 0.0001f + 2.23456f;
    
    // An unsigned 32-bit integer register is needed to hold the 
    // result of the conversion, which packs two 16-bit floats (f16x2).
    unsigned int packed_h;


    // The loop iterates to simulate a heavy workload of this specific instruction.
    for (int i = 0; i < iters; i++) {
        // PTX instruction: cvt.rn.f16x2.f32 Dst_Reg, Src_Reg;
        // This converts the single-precision float in Src_Reg into two half-precision 
        // floats, packs them into the 32-bit register Dst_Reg, and uses 
        // Round-to-Nearest-Even (rn) mode.
        // Since we only provide one f32 input, the instruction sets BOTH f16x2 
        // components (the low and high half-words) to the converted value.
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;"
                      // Output: The 32-bit result register (packed_h) is modified.
                      : "=r"(packed_h)
                      // Input: The single-precision float register (input_f32) is read.
                      : "f"(x), "f"(y));
        
        // To prevent the compiler from optimizing the entire loop away 
        // (since input_f32 is constant), we can use the result as the 
        // input for the next iteration by converting the packed result 
        // back to an f32. This simulates a dependency chain and prevents 
        // dead-code elimination.
        // However, a simpler way is to just write the result to a variable 
        // which prevents dead code elimination since the variable is used 
        // at the end. For purity, we will stick to the basic benchmark 
        // and rely on the volatile keyword.
    }
    
    // Store the final packed half-precision result. 
    // For simplicity, we cast the packed 32-bit unsigned int to a float pointer 
    // and write to the output array. The user's Python code will need to 
    // unpack this result to verify correctness/timing.
    *((unsigned int*)out + threadIdx.x) = packed_h;
}
"""

CUDA_SRC_SHFL="""
extern "C" __global__
void bench_shfl(int *out, int iters) {
    int lane = threadIdx.x & 31;
    int x = lane;
    int mask_param = 2;

    // Force all 32 threads into the same warp
    //if (threadIdx.x >= 32) return;

    for (int i = 0; i < 2*iters; i++) {
        asm volatile(
            "shfl.sync.bfly.b32 %0, %1, %2, 31, -1;"
            : "=r"(x)
            : "r"(x), "r"(mask_param)
        );
    }

    // Store final value so the compiler can't optimize the loop away
    out[threadIdx.x] = x;
}
"""

# --- Build a CUDA-ready Modal image ---
#image = (
#    Image.from_registry("nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.10")
#    .run_commands(
#        "apt update && apt install -y build-essential",
#        "pip install numpy"
#    )
#    .write_file("/workspace/bench.cu", CUDA_SRC)
#)
image = (
        Image.debian_slim(python_version="3.12")
        # 2. Install tools required for CUDA repo
        .apt_install("wget", "curl", "gnupg")

        # 3. Add NVIDIA CUDA repository for Debian 12 (Bookworm)
        .run_commands(
            "wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb",
            "dpkg -i cuda-keyring_1.1-1_all.deb",
            "apt-get update"
        )

        # 4. Install CUDA debugging tools
        .apt_install(
            "cuda-toolkit-12-4",   # contains cuda-gdb
            # OR if you only want debugger:
            # "cuda-gdb",
        )
        .pip_install(
            "numpy",
            "cupy-cuda12x",
            "nvidia-ml-py",
        )
        .workdir("/workspace")
        #.write_file("/workspace/bench.cu", CUDA_SRC)
        .add_local_file("bench.cu", remote_path="/workspace/bench.cu", copy=False) # copy after image build, no need rebuild
)

volume = Volume.from_name(f"{app_name}-dump", create_if_missing=True) # create a cloud volume to store compiled dump files

"""
script

TMPDIR=/ssd2/zhangzn/txl/docker/tmp ncu --metrics smsp__inst_executed_pipe_fma.sum python bench_ptx.py
TMPDIR=/ssd2/zhangzn/txl/docker/tmp ncu --metrics sm__pipe_fma_cycles_active.sum python bench_ptx.py
TMPDIR=/ssd2/zhangzn/txl/docker/tmp ncu --metrics sm__pipe_alu_cycles_active.sum python bench_ptx.py
"""
def main(check_gpu=False):
    import numpy as np
    import ctypes
    import os

    def get_gpu_type():
        import subprocess

        try:
            #result = subprocess.run(['find', '/', '-name', 'libcublas.so*'], capture_output=True, text=True, check=True)
            #output = result.stdout
            #print(output)
            #result = subprocess.run(['find', '/', '-name', 'cuda-gdb'], capture_output=True, text=True, check=True)
            #output = result.stdout
            #print(output)
            #result = subprocess.run(['find', '/', '-name', 'nvcc'], capture_output=True, text=True, check=True)
            #output = result.stdout
            #print(output)

            # Execute nvidia-smi command to query GPU details
            result = subprocess.run(['nvidia-smi', '-q'], capture_output=True, text=True, check=True)
            output = result.stdout

            # Look for indicators of SXM or PCIe in the output
            for line in output.split("\n"):
                if "Product Name" in line:
                    print(line)
                    if 'H100' in line and 'HBM3' in line:
                    #if 'B200' in line:
                        return True
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi: {e}")
        except FileNotFoundError:
            print("nvidia-smi not found. Please ensure NVIDIA drivers are installed and in your PATH.")
        return False

    import pynvml
    def get_clock(handle, use_smi=False):
        """Helper to fetch the current graphics clock speed."""
        # 0 = Graphics Clock, 1 = Memory Clock
        if use_smi:
            active_clock = nvidia_smi.nvmlDeviceGetClockInfo(handle, nvidia_smi.NVML_CLOCK_SM)
        else:
            active_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        print(f"Clock speed DURING workload: {active_clock} MHz")
        return active_clock

    def init_clock(use_smi=False):
        # 1. Initialize NVML and get GPU handle
        try:
            if use_smi:
                nvidia_smi.nvmlInit()
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            else:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                print(f"Monitoring Device: {gpu_name}")
            return handle
        except Exception as e:
            print(f"Error initializing NVML: {e}")
            return

    def close_clock(use_smi=False):
        # 3. Cleanup
        if use_smi:
            nvidia_smi.nvmlShutdown()
        else:
            pynvml.nvmlShutdown()

    if check_gpu and not get_gpu_type():
        return

    # --- Benchmark parameters ---
    #N_THREADS = 1024*1024*4
    #N_THREADS = 128

    ############
    # ctypes
    ############

    ## --- Compile CUDA ---
    #print("Compiling CUDA kernels...")
    #flags = "--shared --compiler-options '-fPIC' -Xcompiler -O3 -gencode arch=compute_90,code=sm_90"
    #subprocess.check_call(
    #    f"/usr/local/cuda-12.4/bin/nvcc /workspace/bench.cu -o /workspace/bench {flags}".split()
    #)
    ## 2. Dump PTX
    #subprocess.check_call(
    #    f"/usr/local/cuda-12.4/bin/nvcc /workspace/bench.cu {flags} -ptx -o /workspace/dump/bench.ptx".split()
    #)
    ## 2. Dump SASS
    #subprocess.check_call(
    #    f"/usr/local/cuda-12.4/bin/cuobjdump --dump-sass /workspace/bench > /workspace/dump/bench.sass", shell=True
    #)
    #bench = ctypes.CDLL("/workspace/bench")
    ## Configure kernel signatures
    #bench.bench_add.argtypes = [ctypes.c_void_p, ctypes.c_int]
    #bench.bench_ex2.argtypes = [ctypes.c_void_p, ctypes.c_int]
    ## Allocate output buffer
    #out = np.zeros(N_THREADS, dtype=np.float32)
    #out_ptr = out.ctypes.data_as(ctypes.c_void_p)

    ############
    # CuPy
    ############

    import cupy as cp

    #mod = cp.RawModule(code=CUDA_SRC_EX2)

    mod = cp.RawModule(code=CUDA_SRC_EX2)
    bench_test = mod.get_function("bench_ex2")
    #mod = cp.RawModule(code=CUDA_SRC_EX22)
    #bench_test = mod.get_function("bench_ex2")
    #mod = cp.RawModule(code=CUDA_SRC_ADD)
    #bench_test = mod.get_function("bench_add")
    #mod = cp.RawModule(code=CUDA_SRC_FMA)
    #bench_test = mod.get_function("bench_fma")
    #mod = cp.RawModule(code=CUDA_SRC_MUL)
    #bench_test = mod.get_function("bench_mul")
    #mod = cp.RawModule(code=CUDA_SRC_CVT)
    #bench_test = mod.get_function("bench_cvt")
    #mod = cp.RawModule(code=CUDA_SRC_SHFL)
    #bench_test = mod.get_function("bench_shfl")

    ITERS = 10000000
    #ITERS = 10
    num_regs = 1
    threads_per_block = 128
    actual_threads_per_block = 128
    #blocks = (N_THREADS + threads_per_block - 1) // threads_per_block
    num_blocks = 1

    out_ptr = cp.zeros(num_regs * threads_per_block, dtype=cp.float32)
    in1_ptr = cp.zeros(num_regs * threads_per_block, dtype=cp.float32)
    in2_ptr = cp.zeros(num_regs * threads_per_block, dtype=cp.float32)

    class Bench:
        pass
    bench=Bench()
    bench.bench_test = lambda out_param, ITERS_param: bench_test((num_blocks,), (actual_threads_per_block,), (out_param, ITERS_param))
    #bench.bench_test = lambda out_param, ITERS_param: bench_test((num_blocks,), (threads_per_block,), (out_param, in1_ptr, in2_ptr, ITERS_param))

    #mod.save("bench.cubin")
    # Now dump SASS
    #import subprocess
    #subprocess.check_call("cuobjdump --dump-sass bench.cubin > /workspace/dump/bench.sass", shell=True)



    def time_kernel(func):
        #start = time.time()
        # Use CUDA events for accurate timing
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        start_event.record()
        func(out_ptr, ITERS)
        end_event.record()
        end_event.synchronize()

        # Time in milliseconds
        elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
        return elapsed_ms/1000

    ## --- Run add.f32 benchmark ---
    #print("\nRunning add.f32...")
    #t_add = time_kernel(bench.bench_add)
    #total_add_ops = N_THREADS * ITERS * 1  # 1 FLOP per add
    #gflops_add = total_add_ops / t_add / 1e9
    #print(f"Time: {t_add:.4f}s")
    #print(f"Throughput: {gflops_add:.2f} GFLOPs")

    # --- Run ex2.approx.f32 benchmark ---
    handle = init_clock()
    print("\nRunning test...")
    t_test = time_kernel(bench.bench_test)
    get_clock(handle)
    close_clock()
    #total_ex2 = N_THREADS * ITERS
    total_test = num_regs * actual_threads_per_block * num_blocks * ITERS
    instr_per_sec = total_test / t_test / 1e12

    lat = (t_test * 1.9) / (num_regs * ITERS) * 1e9

    print(f"Time: {t_test:.4f}s")
    print(f"Total ops: {total_test} ops")
    print(f"Latency-Throughput: {instr_per_sec:.4f} TAIPS/s")
    print(f"Latency: {lat:.4f} cycles/instr")

    print("\nDone.")

@app.function(
    image=image,
    gpu='H100',   # You can switch to A100 / H100 if your account supports it.
    volumes={"/workspace/dump": volume},
    timeout=600
)
def run_bench():
    main(True)

if __name__ == "__main__":
    main()
