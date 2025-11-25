export CUDA_VISIBLE_DEVICES=1
export TRITON_ALWAYS_COMPILE=1

CMD=$1 # ncu, nsys

REGEX="flash|txl|coopA|softmax_kernel"

#OUTPUT="txl_fa_profile"
### 3 levels of metrics

#ANALYSIS="full"
#ANALYSIS="detailed"

#SECTIONS="Occupancy"
#SECTIONS="MemoryWorkloadAnalysis,ComputeWorkloadAnalysis"
#SECTIONS="InstructionStats,SpeedOfLight"
#SECTIONS="WorkloadDistribution"
#SECTIONS="SourceCounters"
#SECTIONS="WarpStateStats" # stalls
#SECTIONS="LaunchStats" # spills

#METRICS="smsp__thread_inst_executed.sum" # quite basic
#METRICS="smsp__pcsamp_warps_issue_stalled_barrier,smsp__pcsamp_warps_issue_stalled_branch_resolving" # not tested

# NOTE: instr count + cycles active
#METRICS="sm__inst_executed_pipe_tensor.sum,sm__inst_executed_pipe_tma.sum" # TODO: what pipe mean?
#METRICS="sm__sass_inst_executed_op_tma.sum" # more accurate
#METRICS="smsp__average_inst_executed_pipe_tma_per_warp"
#METRICS="sm__cycles_active"
#METRICS="sm__pipe_tma_cycles_active"
#METRICS="sm__pipe_tensor_cycles_active"
# NOTE: check stalls
#METRICS="smsp__average_warp_latency_issue_stalled_gmma"
#METRICS="smsp__average_warp_latency_issue_stalled_barrier"
#METRICS="smsp__average_warp_latency_issue_stalled_gmma,smsp__average_warp_latency_issue_stalled_barrier,smsp__average_warp_latency_issue_stalled_long_scoreboard"
#METRICS="smsp__warp_issue_stalled_barrier_per_warp_active" # interesting
#METRICS="smsp__warp_issue_stalled_gmma_per_warp_active" # mbar.arrive not significant
#METRICS="smsp__warp_issue_stalled_wait_per_warp_active" # can be large
#METRICS="smsp__warps_issue_stalled_long_scoreboard"
# spill
#METRICS="l1tex__t_bytes_pipe_lsu_mem_local_op_ld,l1tex__t_bytes_pipe_lsu_mem_local_op_st"
#METRICS="sm__sass_data_bytes_mem_local_op_ld,sm__sass_data_bytes_mem_local_op_st"
# denormal
METRICS="sm__sass_thread_inst_executed_op_fp16_pred_on,sm__sass_thread_inst_executed_op_fp32_pred_on"


PY_SCRIPT=python/txl/tutorials/02-flash-attention.py 
#PY_SCRIPT=python/txl/tutorials/04-softmax.py
#PY_SCRIPT=python/txl/tests/wgid.py
#PY_SCRIPT="python/txl/tutorials/01-matmul.py -K 16384"
#PY_SCRIPT="python/txl/tutorials/01-matmul.py -K 2048"

# Convert comma-separated sections to multiple --section flags
section_flags=()
if [ -n "$SECTIONS" ]; then
    IFS=',' read -ra sections <<< "$SECTIONS"
    for section in "${sections[@]}"; do
        section_flags+=("--section" "$section")
    done
fi

if [ "$CMD" == "ncu" ]; then
    ncu \
        ${METRICS:+--metrics "$METRICS"} \
        ${ANALYSIS:+--set "$ANALYSIS"} \
        "${section_flags[@]}" \
        ${REGEX:+--kernel-name "regex:$REGEX"} \
        ${OUTPUT:+-o "$OUTPUT"} \
        python $PY_SCRIPT
elif [ "$CMD" == "nsys" ]; then
    nsys profile \
        -t cuda \
        ${OUTPUT:+-o "$OUTPUT"} \
        python $PY_SCRIPT
else
    echo "Usage: $0 [ncu|nsys]"
    exit 1
fi
