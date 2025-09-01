from modal import Image, App

app = App(name="txl")  # Note: this is optional since Modal 0.57


txl_image = (
    Image.from_dockerfile(path="./Dockerfile")
    #Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.5.1",
        "numpy==2.1.3",
        "matplotlib",
        "pandas",
        "llnl-hatchet",
        "einops",
    )
    #.add_local_file("requirements.txt", "/home/conda/requirements.txt")
    #.add_local_file("txl-3.4.0-cp312-cp312-linux_x86_64.whl", "/root/txl.whl")
    #.add_local_file("02-flash-attention.py", "/home/conda/02-flash-attention.py")
    .run_commands(
    #    "pip install -r /home/conda/requirements.txt",
    #    "pip uninstall triton",
        "pip install /root/txl-3.4.0-cp312-cp312-linux_x86_64.whl",
    #    "pip install /home/conda/02-flash-attention.py",
    )
)

# Example function that uses the image
@app.function(gpu="H100", image=txl_image)
def test_flash_attention():

    def RUN(cmd):
        import subprocess
    
    
        # Execute the bash script
        result = subprocess.run(["/bin/bash", "-c", cmd], 
                              capture_output=True, text=True, cwd="/home/conda")
        
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

    def get_gpu_type():
        import subprocess
    
        try:
            # Execute nvidia-smi command to query GPU details
            result = subprocess.run(['nvidia-smi', '-q'], capture_output=True, text=True, check=True)
            output = result.stdout

            # Look for indicators of SXM or PCIe in the output
            for line in output.split("\n"):
                if "Product Name" in line:
                    print(line)
                    if 'H100' in line and 'HBM3' in line:
                        return True
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi: {e}")
        except FileNotFoundError:
            print("nvidia-smi not found. Please ensure NVIDIA drivers are installed and in your PATH.")
        return False

    def test_torch():
        import torch
        x= torch.randn(100, 100, device="cuda")
        x = x+x
        x[0].cpu()

    def test_txl():
        import torch
        import os

        import triton
        import triton.language as tl

        try:
            import txl
            Has_TXL = True
            from triton.tools.tensor_descriptor import TensorDescriptor
            DEVICE = triton.runtime.driver.active.get_active_torch_device()

            print("TXL")
        except Exception as e:
            print("NO TXL")
            print(e)

        tma_ws_best_config = {'BLOCK_M':128, 'BLOCK_N':128, 'NUM_CONSUMERS': 2, 'NUM_STAGES': 2} # stages: 3, num warps: 4, num_warpgroups: 3

        def _host_descriptor_pre_hook(nargs):
            NUM_CONSUMER_GROUPS = nargs.get("NUM_CONSUMERS", 1)
            BLOCK_M = nargs["BLOCK_M"] // NUM_CONSUMER_GROUPS
            BLOCK_N = nargs["BLOCK_N"]
            HEAD_DIM = nargs["HEAD_DIM"]
            if not isinstance(nargs["desc_q"], TensorDescriptor):
                return
            nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
            nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
            nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
            nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]

        @triton.jit
        def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
            if isinstance(desc_or_ptr, tl.tensor_descriptor):
                return desc_or_ptr
            else:
                return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)

        @txl.autotune(
            configs=[
                txl.Config(
                    tma_ws_best_config,
                    num_stages=2,
                    num_warps=4,
                    num_warpgroups=3,
                    pre_hook = _host_descriptor_pre_hook,
                )
            ],
            key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT"],
         )
        #@txl.jit(src_file='dump/BFTFLZXGNHJ5H24JXMORYIYU7P6KTCTQ5YJ5EZJSF77MVLWKGJNA/_attn_fwd_ws_tma_txl3.ptx')
        @txl.jit
        def _attn_fwd_ws_tma_txl3(sm_scale, M,  #
                      Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #
                      HEAD_DIM: tl.constexpr,  #
                      BLOCK_M: tl.constexpr,  #
                      BLOCK_N: tl.constexpr,  #
                      FP8_OUTPUT: tl.constexpr,  #
                      STAGE: tl.constexpr,  #
                      warp_specialize: tl.constexpr,  #

                      # NOTE: txl
                      NUM_STAGES: tl.constexpr,  #
                      NUM_CONSUMERS: tl.constexpr  #
                      ):
            dtype = tl.float8e5 if FP8_OUTPUT else tl.float16
            tl.static_assert(BLOCK_N <= HEAD_DIM)
            start_m = tl.program_id(0)
            off_hz = tl.program_id(1)
            off_z = off_hz // H
            off_h = off_hz % H

            y_dim = Z * H * N_CTX
            # If no host desc, then make device desc
            desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                             block_shape=[BLOCK_M, HEAD_DIM])
            desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                             block_shape=[BLOCK_N, HEAD_DIM])
            desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                             block_shape=[BLOCK_N, HEAD_DIM])
            desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                             block_shape=[BLOCK_M, HEAD_DIM])

            offset_y = off_z * (N_CTX * H) + off_h * N_CTX
            qo_offset_y = offset_y + start_m * BLOCK_M


            # load q: it will stay in SRAM throughout
            #q = desc_q.load([qo_offset_y, 0])
            bQ0 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype) # bQ has only 1 buffer for reuse only
            pMbar_bQ0 = txl.mbar_alloc(1)
            bQ1 = txl.smem_alloc([BLOCK_M//2, HEAD_DIM], dtype=dtype)
            pMbar_bQ1 = txl.mbar_alloc(1)

            bK = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
            bV = txl.smem_alloc([BLOCK_N, HEAD_DIM], dtype=dtype, num_stages=NUM_STAGES)
            pMbar_bK = txl.mbar_alloc(1, num_stages=NUM_STAGES)
            pMbar_bV = txl.mbar_alloc(1, num_stages=NUM_STAGES)

            cMbar_QK1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
            cMbar_PV1 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
            cMbar_QK2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)
            cMbar_PV2 = txl.mbar_alloc(128, num_stages=NUM_STAGES)

            #WG1_BAR = 8
            #WG2_BAR = 9
            #WG_NUM_THREADS = 128 * 2

            # TODO: func type mismatch

            # range of values handled by this stage
            lo, hi = 0, N_CTX
            offsetkv_y = offset_y + lo


            if txl.is_warpgroup([0]):

                bQ0i = txl.get_buffer(bQ0, 0)
                pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
                bQ1i = txl.get_buffer(bQ1, 0)
                pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)

                txl.mbar_expect(pMbar_bQ0i, BLOCK_M // 2 * HEAD_DIM * 2)
                txl.tma_load(bQ0i, desc_q, [qo_offset_y, 0], pMbar_bQ0i)
                txl.mbar_wait(pMbar_bQ0i, 0)
                txl.mbar_expect(pMbar_bQ1i, BLOCK_M // 2 * HEAD_DIM * 2)
                txl.tma_load(bQ1i, desc_q, [qo_offset_y+BLOCK_M//2, 0], pMbar_bQ1i)
                txl.mbar_wait(pMbar_bQ1i, 0)

                bufIdxW = 0 # write buffer
                phase = 1

                for start_n in range(lo, hi, BLOCK_N):
                    start_n = tl.multiple_of(start_n, BLOCK_N)
                    cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxW)
                    cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxW)
                    cur_bK = txl.get_buffer(bK, bufIdxW)
                    cur_bV = txl.get_buffer(bV, bufIdxW)

                    cur_mbar_QK1 = txl.get_buffer(cMbar_QK1, bufIdxW) # wait for the same buffer
                    cur_mbar_PV1 = txl.get_buffer(cMbar_PV1, bufIdxW)
                    cur_mbar_QK2 = txl.get_buffer(cMbar_QK2, bufIdxW)
                    cur_mbar_PV2 = txl.get_buffer(cMbar_PV2, bufIdxW)

                    # TODO: tma_expect_and_load
                    txl.mbar_wait(cur_mbar_QK1, phase)
                    txl.mbar_wait(cur_mbar_QK2, phase)
                    txl.mbar_expect(cur_mbar_bK, BLOCK_N * HEAD_DIM * 2)
                    txl.tma_load(cur_bK, desc_k, [offsetkv_y, 0], cur_mbar_bK)

                    txl.mbar_wait(cur_mbar_PV1, phase)
                    txl.mbar_wait(cur_mbar_PV2, phase)
                    txl.mbar_expect(cur_mbar_bV, BLOCK_N * HEAD_DIM * 2)
                    txl.tma_load(cur_bV, desc_v, [offsetkv_y, 0], cur_mbar_bV)

                    offsetkv_y += BLOCK_N
                    bufIdxW = (bufIdxW + 1) % NUM_STAGES
                    if bufIdxW == 0:
                        phase = phase^1


            if txl.is_warpgroup([1, 2]):

                if txl.is_warpgroup([1]):
                    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M//2)

                    # first let wg1 to start
                    #txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)
                    txl.bar_arrive(8, 256)
                else:
                    offs_m = start_m * BLOCK_M + tl.arange(BLOCK_M//2, BLOCK_M)

                # initialize pointer to m and l
                # These are in regs
                m_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) - float("inf")
                l_i = tl.zeros([BLOCK_M//2], dtype=tl.float32) + 1.0
                acc = tl.zeros([BLOCK_M//2, HEAD_DIM], dtype=tl.float32)
                # load scales
                qk_scale = sm_scale
                qk_scale *= 1.44269504  # 1/log(2)

                ## load and wait Q
                bQ0i = txl.get_buffer(bQ0, 0)
                pMbar_bQ0i = txl.get_buffer(pMbar_bQ0, 0)
                bQ1i = txl.get_buffer(bQ1, 0)
                pMbar_bQ1i = txl.get_buffer(pMbar_bQ1, 0)

                if txl.is_warpgroup([1]):
                    txl.mbar_wait(pMbar_bQ0i, 0)
                    # WG1 just start
                    #txl.bar_wait(WG1_BAR, WG_NUM_THREADS)
                if txl.is_warpgroup([2]):
                    txl.mbar_wait(pMbar_bQ1i, 0)
                    # WG2 start after wg1 gemm0
                    #txl.bar_wait(WG2_BAR, WG_NUM_THREADS)


                # -- prologue --

                # TODO: write in txl.jit for reuse
                ## load and wait K
                cur_mbar_bK = txl.get_buffer(pMbar_bK, 0)
                cur_bK = txl.get_buffer(bK, 0)
                txl.mbar_wait(cur_mbar_bK, 0)

                if txl.is_warpgroup([1]):
                    cur_mbar_QK = txl.get_buffer(cMbar_QK1, 0)
                    qk = tl.dot(bQ0i, cur_bK.T)

                    # TODO whether before dot wait?
                    #txl.bar_arrive(WG2_BAR, WG_NUM_THREADS)
                    txl.dot_wait(0)
                    txl.mbar_arrive(cur_mbar_QK)

                else: # [2]
                    cur_mbar_QK = txl.get_buffer(cMbar_QK2, 0)
                    qk = tl.dot(bQ1i, cur_bK.T)

                    # TODO whether before dot wait?
                    #txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)
                    txl.dot_wait(0)
                    txl.mbar_arrive(cur_mbar_QK)

                m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                qk = qk * qk_scale - m_ij[:, None]

                p = tl.math.exp2(qk)
                l_ij = tl.sum(p, 1)
                # -- update m_i and l_i
                alpha = tl.math.exp2(m_i - m_ij)
                l_i = l_i * alpha + l_ij
                # -- update output accumulator --
                #acc = acc * alpha[:, None]

                # update m_i and l_i
                m_i = m_ij

                # update acc
                p = p.to(dtype)

                bufIdxRK = 1
                bufIdxRV = 0
                phaseK = 0
                phaseV = 0

                # pass: p, l_i, m_i, acc
                # loop over k, v and update accumulator
                for start_n in range(lo+BLOCK_N, hi, BLOCK_N):
                    start_n = tl.multiple_of(start_n, BLOCK_N)

                    # -- load k ----
                    cur_mbar_bK = txl.get_buffer(pMbar_bK, bufIdxRK)
                    cur_bK = txl.get_buffer(bK, bufIdxRK)
                    txl.mbar_wait(cur_mbar_bK, phaseK)

                    # Now only consider gemm 0 and softmax(gemm 1)
                    # --- wait to start gemm 1 ---
                    # case 1: wg1 earlys start
                    # case 2: wait the release from last iter gemm 0 ends?
                    if txl.is_warpgroup([1]):
                        # WG1 just start
                        #txl.bar_wait(WG1_BAR, WG_NUM_THREADS)
                        txl.bar_wait(8, 256)
                    if txl.is_warpgroup([2]):
                        # WG2 start after wg1 gemm0
                        #txl.bar_wait(WG2_BAR, WG_NUM_THREADS)
                        txl.bar_wait(9, 256)

                    if txl.is_warpgroup([1]):
                        cur_mbar_QK = txl.get_buffer(cMbar_QK1, bufIdxRK)
                        cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxRV)
                        qk = tl.dot(bQ0i, cur_bK.T)

                    else: # [2]
                        cur_mbar_QK = txl.get_buffer(cMbar_QK2, bufIdxRK)
                        cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxRV)
                        qk = tl.dot(bQ1i, cur_bK.T)

                    # -- compute pv j-1 ----
                    # load v
                    cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
                    cur_bV = txl.get_buffer(bV, bufIdxRV)
                    txl.mbar_wait(cur_mbar_bV, phaseV)

                    ## Downgrade if put before load v
                    #if txl.is_warpgroup([1]):
                    #    txl.bar_wait(WG1_BAR, WG_NUM_THREADS)
                    #else:
                    #    txl.bar_wait(WG2_BAR, WG_NUM_THREADS)

                    # note that this non transposed v for FP8 is only supported on Blackwell
                    acc = tl.dot(p, cur_bV, acc)

                    txl.dot_wait(1)
                    # TODO: before or after wait? oh previously is also before QK wait
                    if txl.is_warpgroup([1]):
                        #txl.bar_arrive(WG2_BAR, WG_NUM_THREADS)
                        txl.bar_arrive(9, 256)
                    else:
                        #txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)
                        txl.bar_arrive(8, 256)
                    # --- release QK finished ---
                    txl.mbar_arrive(cur_mbar_QK)

                    #m_i, l_i, p, alpha = softmax_txl(m_i, l_i, qk, qk_scale, dtype)

                    # -- compute softamx, block arg updates ----
                    m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
                    qk = qk * qk_scale - m_ij[:, None]

                    # udpate p
                    p = tl.math.exp2(qk)
                    l_ij = tl.sum(p, 1)
                    # update m_i and l_i
                    alpha = tl.math.exp2(m_i - m_ij)
                    l_i = l_i * alpha + l_ij
                    m_i = m_ij

                    # update acc, NOTE: p position is important
                    p = p.to(dtype)

                    # update output accumulator
                    txl.dot_wait(0)
                    # --- release PV j-1 finished ---
                    txl.mbar_arrive(cur_mbar_PV)

                    acc = acc * alpha[:, None]

                    bufIdxRK = (bufIdxRK + 1) % NUM_STAGES
                    if bufIdxRK == 0:
                        phaseK = phaseK ^ 1
                    bufIdxRV = (bufIdxRV + 1) % NUM_STAGES
                    if bufIdxRV == 0:
                        phaseV = phaseV ^ 1


                #if txl.is_warpgroup([1]):
                #    txl.bar_arrive(WG2_BAR, WG_NUM_THREADS)
                #else:
                #    txl.bar_arrive(WG1_BAR, WG_NUM_THREADS)

                # -- last iter --
                # load v
                cur_mbar_bV = txl.get_buffer(pMbar_bV, bufIdxRV)
                #if txl.is_warpgroup([1]):
                #    cur_mbar_PV = txl.get_buffer(cMbar_PV1, bufIdxRV)
                #else:
                #    cur_mbar_PV = txl.get_buffer(cMbar_PV2, bufIdxRV)
                cur_bV = txl.get_buffer(bV, bufIdxRV)
                txl.mbar_wait(cur_mbar_bV, phaseV)

                # note that this non transposed v for FP8 is only supported on Blackwell
                acc = tl.dot(p, cur_bV, acc)
                txl.dot_wait(0)
                #txl.mbar_arrive(cur_mbar_PV)

                # epilogue
                m_i += tl.math.log2(l_i)
                acc = acc / l_i[:, None]
                m_ptrs = M + off_hz * N_CTX + offs_m
                tl.store(m_ptrs, m_i)

                if txl.is_warpgroup([1]):
                    desc_o.store([qo_offset_y, 0], acc.to(dtype))
                if txl.is_warpgroup([2]):
                    desc_o.store([qo_offset_y+BLOCK_M//2, 0], acc.to(dtype))

        def is_hip():
            return triton.runtime.driver.active.get_current_target().backend == "hip"


        def is_cuda():
            return triton.runtime.driver.active.get_current_target().backend == "cuda"


        def supports_host_descriptor():
            return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


        def is_blackwell():
            return is_cuda() and torch.cuda.get_device_capability()[0] == 10

        class _attention(torch.autograd.Function):

            @staticmethod
            def forward(ctx, q, k, v, causal, sm_scale, algo=0, no_tune=False, profiling=False):
                # shape constraints
                HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
                # when v is in float8_e5m2 it is transposed.
                HEAD_DIM_V = v.shape[-1]
                assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
                assert HEAD_DIM_K in {16, 32, 64, 128, 256}
                o = torch.empty_like(q)
                stage = 3 if causal else 1
                extra_kern_args = {}
                # Tuning for AMD target

                M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
                if supports_host_descriptor():
                    # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
                    y_dim = q.shape[0] * q.shape[1] * q.shape[2]

                    dummy_block = [1, 1]
                    desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                    desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                    desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                    desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
                else:
                    desc_q = q
                    desc_v = v
                    desc_k = k
                    desc_o = o

                def alloc_fn(size: int, align: int, _):
                    return torch.empty(size, dtype=torch.int8, device="cuda")

                triton.set_allocator(alloc_fn)

                def grid(META):
                    return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

                ctx.grid = grid
                algo_map = {
                    #0: _attn_fwd_tma_txl,
                    #1: _attn_fwd_ws_tma_txl1,
                    #2: _attn_fwd_ws_tma_txl2,
                    3: _attn_fwd_ws_tma_txl3,
                    #4: _attn_fwd_ws_tma_txl4,
                }

                if profiling:
                    proton.start("fa3", backend="instrumentation", mode='default:sampling_strategy=selective:sampling_options=0,4,8', data="trace")
                algo_map[algo][grid](
                    sm_scale, M,  #
                    q.shape[0], q.shape[1],  #
                    desc_q, desc_k, desc_v, desc_o,  #
                    N_CTX=q.shape[2],  #
                    HEAD_DIM=HEAD_DIM_K,  #
                    FP8_OUTPUT=q.dtype == torch.float8_e5m2,  #
                    STAGE=stage,  #
                    warp_specialize=False,  #
                    **extra_kern_args)

                if profiling:
                    proton.finalize()

                ctx.save_for_backward(q, k, v, o, M)
                ctx.sm_scale = sm_scale
                ctx.HEAD_DIM = HEAD_DIM_K
                ctx.causal = causal
                return o
        attention = _attention.apply
        HAS_FLASH=False

        import sys
        import math
        from txl.tests.test_util import attention_ref
        def test_op(Z, H, N_CTX, HEAD_DIM, causal, dtype=torch.float16, algo=0, no_tune=False, profiling=False):
            q = (torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE))
            k = (torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE))
            v = (torch.randn((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE))
            q1 = q.permute(0,2,1,3).contiguous()
            k1 = k.permute(0,2,1,3).contiguous()
            v1 = v.permute(0,2,1,3).contiguous()

            # txl
            if HAS_FLASH:
                if PYFLASH:
                    tri_out, lse = flash_attn_func(q1, k1, v1, causal=causal)
                    tri_out = tri_out.half()
                else:
                    tri_out = flash_attn_func(q1, k1, v1, causal=causal).half()
                tri_out = tri_out.permute(0,2,1,3).contiguous()
            elif Has_TXL:
                tri_out = attention(q, k, v, causal, 1/math.sqrt(HEAD_DIM), algo, no_tune, profiling).half()

            if profiling:
                exit()
            ref_out, ref_attn = attention_ref(q1, k1, v1, causal=causal)
            ref_out = ref_out.permute(0,2,1,3).contiguous()
            print(f"Output max diff: {(tri_out - ref_out).abs().max().item()}")
            print(f"Output mean diff: {(tri_out - ref_out).abs().mean().item()}")
            assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)

        TORCH_HAS_FP8 = hasattr(torch, 'float8_e5m2')
        BATCH, N_HEADS, HEAD_DIM = 16, 32, 128

        TORCH_HAS_FP8=False
        # vary seq length for fixed head and batch=4
        configs = []
        for mode in ["fwd"]:
            for causal in [False]:
                for warp_specialize in [False, True] if is_blackwell() else [False]:
                    if mode == "bwd" and not causal:
                        continue
                    configs.append(
                        triton.testing.Benchmark(
                            x_names=["N_CTX"],
                            x_vals=[2**i for i in range(10, 15)],
                            line_arg="provider",
                            line_vals=(["triton-fp16"] if Has_TXL else []) + (["triton-fp8"] if TORCH_HAS_FP8 else []) +
                            (["flash"] if HAS_FLASH else []),
                            line_names=(["Triton [FP16]"] if Has_TXL else [])  + (["Triton [FP8]"] if TORCH_HAS_FP8 else []) +
                            (["Flash-3"] if HAS_FLASH else []),
                            styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                            ylabel="TFLOPS",
                            plot_name=
                            f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{mode}-causal={causal}-warp_specialize={warp_specialize}",
                            args={
                                "H": N_HEADS,
                                "BATCH": BATCH,
                                "HEAD_DIM": HEAD_DIM,
                                "mode": mode,
                                "causal": causal,
                            },
                        ))
        @triton.testing.perf_report(configs)
        def bench_flash_attention(BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device=DEVICE, algo=0, no_tune=False):
            # follow fa3 paper
            BATCH = int(16384 / N_CTX)
            assert mode in ["fwd", "bwd"]
            dtype = torch.float16
            q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
            k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
            v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device)
            if "triton" in provider:
                if mode == "fwd" and "fp8" in provider:
                    q = q.to(torch.float8_e5m2)
                    k = k.to(torch.float8_e5m2)
                    v = v.permute(0, 1, 3, 2).contiguous()
                    v = v.permute(0, 1, 3, 2)
                    v = v.to(torch.float8_e5m2)
                sm_scale = 1/math.sqrt(HEAD_DIM)
                fn = lambda: attention(q, k, v, causal, sm_scale, algo, no_tune)
                if mode == "bwd":
                    o = fn()
                    do = torch.randn_like(o)
                    fn = lambda: o.backward(do, retain_graph=True)
                ms = triton.testing.do_bench(fn, warmup=1000, rep=1000)

            if provider == "flash":
                # qkv = torch.randn((BATCH, N_CTX, 3, H, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
                # fn = lambda: flash_attn_func(qkv, causal=causal)
                q = q.permute(0,2,1,3).contiguous()
                k = k.permute(0,2,1,3).contiguous()
                v = v.permute(0,2,1,3).contiguous()
                fn = lambda: flash_attn_func(q, k, v, causal=causal)
                if mode == "bwd":
                    o = fn()
                    do = torch.randn_like(o)
                    fn = lambda: o.backward(do, retain_graph=True)
                ms = triton.testing.do_bench(fn, warmup=1000, rep=1000)

            flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * HEAD_DIM
            total_flops = 2 * flops_per_matmul
            if causal:
                total_flops *= 0.5
            if mode == "bwd":
                total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
            return total_flops * 1e-12 / (ms * 1e-3)

        no_tune=True
        PROFILING=False
        test_op(16, 32, 1024, 128, False, dtype=torch.float16, algo=3, no_tune=no_tune, profiling=PROFILING)
        bench_flash_attention.run(save_path=".", print_data=True, algo=3, no_tune=no_tune)

    test_torch()
    test_txl()

    if not get_gpu_type():
        return
