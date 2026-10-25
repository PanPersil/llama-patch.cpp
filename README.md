# llama-patch.cpp

llama.cpp with the HauhauCS FastMTP patch, a strongly optimized Metal backend and K_P quantization mixes.

## What this is built from

This repository starts from upstream [llama.cpp](https://github.com/ggml-org/llama.cpp) at commit
`4df29be4f4c3673f428170fda944a5b19f743bb8` with the HauhauCS FastMTP patch applied:

```sh
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 4df29be4f4c3673f428170fda944a5b19f743bb8
curl -L -o HauhauCS-FastMTP-llama.cpp.patch \
  https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF/resolve/main/HauhauCS-FastMTP-llama.cpp.patch
git apply --check HauhauCS-FastMTP-llama.cpp.patch
git apply HauhauCS-FastMTP-llama.cpp.patch
```

The upstream commit history is not included. The history was cut after the patch was applied, so the first commit
here is upstream `4df29be` plus the FastMTP patch, and every later commit is a change made in this repository.

## The model

The patch comes with
[HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF](https://huggingface.co/HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF),
an uncensored build of Qwen3.8-27B. The model is a hybrid: 48 gated delta net (linear attention) layers and 16 full
attention layers, 262144 tokens of context, a vision projector and one MTP block for speculative decoding. The repository
has quants from IQ2_M to Q8_K_P and a FastMTP sidecar (903 MB) that works as the draft model with every text quant.

Example server with MTP speculative decoding:

```sh
./build/bin/llama-server \
  --model Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf \
  --spec-draft-model Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-FastMTP-32K.gguf \
  --spec-type draft-mtp --spec-draft-ngl all --spec-draft-n-max 3 --spec-draft-p-min 0.6 \
  --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 \
  --n-gpu-layers 100 --ctx-size 262144 --jinja
```

The server listens on port 11434 by default in this repository.

## Changes compared to upstream + FastMTP

### Metal backend

- Faster mat-vec for Q5_K: the 5th bit is merged into each byte with mask operations and 4 quants are converted at once.
  A layout with 4 row groups per simdgroup shares the activation loads between rows, so extra tokens in a batch cost
  little. This matters for the verify step of speculative decoding (MTP or a draft model), which runs 2 to 5 tokens.
- Small-batch kernels (2 to 8 tokens) for all K-quants and i-quants. Before, these types read the weights once per token
  for small batches.
- Faster Q5_K dequantization for prompt processing, and 64x64 mat-mat tiles on GPUs without the Metal tensor API.
- Gated delta net: the new recurrent state is written straight into the cache (no extra copy), and the single-token
  SSM_CONV kernel runs one row per thread with the following SiLU fused into it.
- IQ4_NL K/V cache support in flash attention. Before, an IQ4_NL KV cache made attention fall back to the CPU.
- `GGML_METAL_PROFILE=1` prints the GPU time of every op (GPU timestamps at encoder boundaries).
  `GGML_METAL_PROFILE=2` reports the GPU time of whole graphs and the CPU time spent encoding them.

All changed ops are covered by `test-backend-ops`, with extra tests for multi-block rows, small batches, IQ4_NL flash
attention and the gated delta net cache fusion.

### K_P quantization mixes

`llama-quantize` has the K_P mixes Q2_K_P, Q3_K_P, Q4_K_P, Q5_K_P, Q6_K_P and Q8_K_P. They give attention, `ffn_down`,
the gated delta net projections, `attn_v`, the output head and the MTP input projection more bits than the base K-quant
and keep `ffn_up` / `ffn_gate` at or below it. They need an imatrix:

```sh
./build/bin/llama-quantize --imatrix imatrix.gguf model-BF16.gguf model-Q4_K_P.gguf Q4_K_P
```

For compatibility the file records the matching standard type (Q2_K, Q3_K_M, Q4_K_M, Q5_K_M, Q6_K or Q8_0).
On Qwen3.8-27B with the MTP block the mixes give the same bits per weight as the published K_P files:

| mix | bpw |
|---|---:|
| Q8_K_P | 9.21 |
| Q6_K_P | 7.59 |
| Q5_K_P | 5.92 |
| Q4_K_P | 5.25 |
| Q3_K_P | 3.93 |
| Q2_K_P | 3.12 |

Q4_K_P gives exactly the tensor types of the published HauhauCS Q4_K_P file.

### Server

- Default port 11434.
- Default model owner `UnionCors`.

## Results

The optimizations made a real difference on my machine (MacBook Pro, M2 Max with a 30-core GPU).
Qwen3.8-27B Q4_K_P, flash attention, q4_0 KV cache, before (upstream + FastMTP) and after:

| llama-bench | before | after |
|---|---:|---:|
| pp512 | 116 t/s | 144 t/s |
| tg64 | 8.3 t/s | 10.0 t/s |
| pp512 at 8k depth | 91 t/s | 109 t/s |
| tg64 at 8k depth | 7.3 t/s | 8.8 t/s |

| batch of tokens (verify step) | before | after |
|---|---:|---:|
| 1 token | 88 ms | 70 ms |
| 2 tokens | 151 ms | 98 ms |
| 3 tokens | 215 ms | 129 ms |
| 4 tokens | 192 ms | 148 ms |

With the FastMTP sidecar on a short code prompt at temperature 0, `--spec-draft-n-max 3 --spec-draft-p-min 0.6` gave
20.3 t/s against 10.5 t/s without a draft (1.93x). The gain depends on the prompt and on how many draft tokens are
accepted.

Perplexity did not change: 3.4454 before and after on the same text with `-c 1024 --chunks 12`.

## License

MIT, see [LICENSE](LICENSE). llama.cpp is by the ggml authors, the FastMTP patch is by HauhauCS.
