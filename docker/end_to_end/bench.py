# TRITON_INTERPRET=1 python3 bench.py

import torch
import triton
import time
from transformers import AutoTokenizer
from gpt import convert_hf_and_load_model, load_model_only

STRING = """\
Large language models have been shown to achieve remarkable performance across a variety of natural\
language tasks using few-shot learning, which drastically reduces the number of task-specific training\
examples needed to adapt the model to a particular application. To further our understanding of the\
impact of scale on few-shot learning, we trained a 540-billion parameter, densely activated, Transformer\
language model, which we call Pathways Language Model (PaLM).\
We trained PaLM on 6144 TPU v4 chips using Pathways, a new ML system which enables highly efficient\
training across multiple TPU Pods. We demonstrate continued benefits of scaling by achieving state-ofthe-art few-shot learning results on hundreds of language understanding and generation benchmarks. On a\
number of these tasks, PaLM 540B achieves breakthrough performance, outperforming the finetuned stateof-the-art on a suite of multi-step reasoning tasks, and outperforming average human performance on the\
recently released BIG-bench benchmark. A significant number of BIG-bench tasks showed discontinuous\
improvements from model scale, meaning that performance steeply increased as we scaled to our largest\
model. PaLM also has strong capabilities in multilingual tasks and source code generation, which we\
demonstrate on a wide array of benchmarks. We additionally provide a comprehensive analysis on bias\
and toxicity, and study the extent of training data memorization with respect to model scale. Finally,\
we discuss the ethical considerations related to large language models and discuss potential mitigation\
strategies.\
"""

def run_benchmark(provider, warmup=25, rep=100, mixed_precison=True, max_length=1024, need_gpt=True):
    assert torch.cuda.is_available()
    device = "cuda"
    model_id = "gpt2"
    if need_gpt:
      model, hf_model, txl_model = convert_hf_and_load_model(model_id, device)
    else:
      model, txl_model = load_model_only(device)
    if mixed_precison:
      model.to(torch.float16)
      txl_model.to(torch.float16)
    # hf_model.to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # triton is slow for batch_size = 1 with current settings but much faster with batch > 1
    inputs = tokenizer([STRING*1000] * 32, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # print(inputs)
    print("input_ids shape:", inputs["input_ids"].shape)
    # exit()
    with torch.no_grad():
      # z_torch = hf_model(**inputs).last_hidden_state
      # z = model(inputs["input_ids"])
      # z_txl = txl_model(inputs["input_ids"])
      # print(f"txl output sample: {z_txl[0, :5, :5]}")
      # print(f"hf_model output sample: {z_torch[0, :5, :5]}")
      # print(f"model output sample: {z[0, :5, :5]}")
      # max_diff = torch.abs(z_torch - z).max().item()
      # max_diff_txl = torch.abs(z_txl - z).max().item()
      # max_diff_txl_hf = torch.abs(z_txl - z_torch).max().item()
      # print(f"max diff between hf_model and model: {max_diff}")
      # print(f"max diff between txl_model and model: {max_diff_txl}")
      # print(f"max diff between txl_model and hf_model: {max_diff_txl_hf}")
      # max_diff_location = torch.abs(z_torch - z).argmax().item()
      # print(f"model output around max diff location: {z.view(-1)[max_diff_location-5:max_diff_location+5]}")
      # print(f"hf_model output around max diff location: {z_torch.view(-1)[max_diff_location-5:max_diff_location+5]}")
      # print(f"txl_model output around max diff location: {z_txl.view(-1)[max_diff_location-5:max_diff_location+5]}")
      # max_diff_location_txl = torch.abs(z_txl - z_torch).argmax().item()
      # print(f"model output around max diff location (txl): {z.view(-1)[max_diff_location_txl-5:max_diff_location_txl+5]}")
      # print(f"txl_model output around max diff location: {z_txl.view(-1)[max_diff_location_txl-5:max_diff_location_txl+5]}")
      # print(f"hf_model output around max diff location (txl): {z_torch.view(-1)[max_diff_location_txl-5:max_diff_location_txl+5]}")
      if provider == "torch":
          def fn():
            if mixed_precison:
              with torch.autocast(device_type="cuda", dtype=torch.float16):
                return hf_model(**inputs).last_hidden_state
            else:
              return hf_model(**inputs).last_hidden_state
          return triton.testing.do_bench(fn, warmup=warmup, rep=rep)
      if provider == "triton":
          fn = lambda: model(inputs["input_ids"])
          return triton.testing.do_bench(fn, warmup=warmup, rep=rep)
      if provider == "txl":
          fn = lambda: txl_model(inputs["input_ids"])
          return triton.testing.do_bench(fn, warmup=warmup, rep=rep)

def run_benchmark_all(warmup=25, rep=100, mixed_precison=True, max_length=1024):
    assert torch.cuda.is_available()
    device = "cuda"
    model_id = "gpt2"
    model, hf_model, txl_model = convert_hf_and_load_model(model_id, device)
    if mixed_precison:
      model.to(torch.float16)
      txl_model.to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer([STRING*1000] * 32, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print("input_ids shape:", inputs["input_ids"].shape)
    with torch.no_grad():
      def fn():
        if mixed_precison:
          with torch.autocast(device_type="cuda", dtype=torch.float16):
            return hf_model(**inputs).last_hidden_state
        else:
          return hf_model(**inputs).last_hidden_state
      torch_time = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
      torch.cuda.synchronize()
      time.sleep(1)  
      fn = lambda: model(inputs["input_ids"])
      triton_time = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
      torch.cuda.synchronize()
      time.sleep(1)  
      fn = lambda: txl_model(inputs["input_ids"])
      txl_time = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
      torch.cuda.synchronize()
      time.sleep(1)  
    return torch_time, triton_time, txl_time

def validate(mixed_precison=True, max_length=1024):
    assert torch.cuda.is_available()
    device = "cuda"
    model_id = "gpt2"
    model, hf_model, txl_model = convert_hf_and_load_model(model_id, device)
    if mixed_precison:
      model.to(torch.float16)
      txl_model.to(torch.float16)
    # hf_model.to(torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # triton is slow for batch_size = 1 with current settings but much faster with batch > 1
    inputs = tokenizer([STRING*1000] * 32, return_tensors="pt", max_length=max_length, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    print(inputs)
    print("input_ids shape:", inputs["input_ids"].shape)
    # exit()
    with torch.no_grad():
      z_torch = hf_model(**inputs).last_hidden_state
      z = model(inputs["input_ids"])
      z_txl = txl_model(inputs["input_ids"])
      print(f"txl output sample: {z_txl[0, :5, :5]}")
      print(f"hf_model output sample: {z_torch[0, :5, :5]}")
      print(f"triton_model output sample: {z[0, :5, :5]}")
      mean_diff_triton = torch.abs(z_torch - z).mean().item()
      mean_diff_txl = torch.abs(z_torch - z_txl).mean().item()
      print(f"mean diff between triton_model and hf_model: {mean_diff_triton}")
      print(f"mean diff between txl_model and hf_model: {mean_diff_txl}")

# 1 A100 40 GB
# torch: batch_size = 512 && t = 1801.32
# triton: batch_size = 512 && t = 789.14
# torch: batch_size = 1024 && OOM
# triton: batch_size = 2048 && t = 3153.70

print("start")
# validate(mixed_precison=True, max_length=1024)
# print("triton:", run_benchmark("triton"))
# print("txl:", run_benchmark("txl"))
# print("torch:", run_benchmark("torch"))
# print("triton long:", run_benchmark("triton", max_length=4096, need_gpt=False))
# print("txl long:", run_benchmark("txl", max_length=4096, need_gpt=False))
for i in range(8, 13):
    length = 2**i
    torch_time , triton_time, txl_time = run_benchmark_all(max_length=length)
    speedup_triton = torch_time / triton_time
    speedup_txl = torch_time / txl_time
    print(f"length: {length}, torch: {torch_time:.2f} ms, triton: {triton_time:.2f} ms, txl: {txl_time:.2f} ms, speedup_triton: {speedup_triton:.2f}x, speedup_txl: {speedup_txl:.2f}x")
# for i in range(11, 13):
#     length = 2**i
#     triton_time = run_benchmark("triton", max_length=length, need_gpt=False)
#     txl_time = run_benchmark("txl", max_length=length, need_gpt=False)
#     speedup_txl = triton_time / txl_time
#     print(f"length: {length}, triton: {triton_time:.2f} ms, txl: {txl_time:.2f} ms, speedup_txl: {speedup_txl:.2f}x")
# print("txl test:", run_benchmark("txl", max_length=1024, need_gpt=False))
# print("triton test:", run_benchmark("triton", max_length=1024, need_gpt=False))
# OLD SUMMARY
# fp32
# torch: 1800
# triton: 789.14

# mixed precision
# torch: 510.80
# triton: 429.80

# fp16
# torch: 301.92

# triton with mixed precison = False
# ffn cast enabled: 791.13
# flash cast enabled: 759.71
# num_warps = 8 & BLOCK_SIZE = 64 ffn :: 759.18
# num_warps = 8 & BLOCK_SIZE = 128 ffn :: 463.80
# layer norm BLOCK_SIZE = 32768 :: 832.63
# layer norm BLOCK_SIZE = 512 :: 462.61
# embeddings BLOCK_SIZE = 512 :: 462.87
# attention BLOCK_SIZE = 128 & num_stages = 4 :: 1279.38
# attention BLOCK_SIZE = 128 & num_stages = 8 :: 460.27
# final config: embeddings (512, 4) + layer norm (512, 4) + ffn (128, 128, 64, 8) + attention (128, 8)

# mixed precision = True
# triton: 273.61
# with attention (128, 8), t = 900 but with attention (64, 4), t = 273!

# mixed precision = False
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch: 623.3262329101562

