# Survey on Efficient Inference and Small Language Models

## TL;DR

- FlashAttention introduced a foundational IO-aware exact attention algorithm that scales Transformer inference efficiently by reducing memory access costs on GPUs, enabling up to 3x speedup and longer context handling [1].
- vLLM PagedAttention advances runtime memory management for KV cache, partitioning it into fixed-size blocks inspired by virtual memory paging, resulting in 2-4x throughput improvements during inference serving [2][3].
- Post-training quantization methods AWQ and GPTQ enable efficient small model inference with 4-bit weights by minimizing quantization error and preserving accuracy, facilitating deployment on resource-constrained hardware [4][5][6][7].
- Speculative decoding and adaptive sampling approaches leverage small draft models with multi-token proposals verified by larger target models, achieving up to 6.5x inference speedups while preserving output quality [8][9][10][11].
- Theoretical advances in compositional decoding frameworks provide principled approaches to balancing sample quality and diversity, presenting future directions for improved decoding efficiency [12].

## Background

Transformers underpin most large language models (LLMs) but their self-attention has quadratic memory and compute complexity with sequence length, limiting efficient inference especially on resource-constrained hardware. Efficient inference techniques seek to reduce memory and compute bottlenecks while preserving output accuracy. Small language models, including reduced-parameter or quantized LMs, enable deployment on edge devices and low-cost inference platforms.

Core technical challenges include optimizing attention computation to reduce memory bandwidth utilization, minimizing latency by improved memory management or decoding algorithms, and compressing model weights with minimal accuracy loss. Foundational works such as FlashAttention demonstrate algorithmic restructuring to minimize GPU memory IO, while recent system innovations like vLLM PagedAttention improve runtime efficiency for multi-request serving scenarios. Quantization methods such as AWQ and GPTQ provide post-training compression frameworks enabling 4-bit weight quantization in small to large models. Sampling and decoding strategies leverage lightweight speculative models and reinforcement learning to reduce inference time without compromising output quality.

## Efficient Attention Computation: FlashAttention and vLLM PagedAttention

FlashAttention [1] is a pioneering technique addressing the quadratic complexity of self-attention by restructuring attention as an IO-aware tiling problem. The method splits input sequences into blocks processed in GPU on-chip SRAM, reducing the number of reads and writes to slower GPU device memory (high bandwidth memory or HBM). By fusing the entire attention operation into a single CUDA kernel and incrementally computing the softmax normalization, FlashAttention reduces memory bandwidth bottlenecks and achieves linear memory scaling with sequence length.

Empirical results demonstrate FlashAttention achieves a 3x inference speedup on GPT-2 with 1K sequence length and supports sequence lengths up to 64K tokens, enabling longer context windows for Transformer models. Its recomputation of partial results trades some increase in computation for large memory savings, beneficial in latency-sensitive inference. FlashAttention has been integrated into multiple production frameworks and is foundational for efficient transformer implementations.

vLLM's PagedAttention [2][3] builds on a system-level memory management concept inspired by OS virtual memory. Unlike FlashAttention’s kernel-level fusion, PagedAttention partitions the key-value (KV) cache into fixed-size blocks allowing non-contiguous and dynamically allocated memory layouts, which reduces fragmentation and enables efficient sharing across parallel requests. The kernel design optimizes memory loading and computation via threads organized into warps to coalesce memory access and supports stable normalization via parallel reduction.

Implemented as part of vLLM, PagedAttention yields 2-4x throughput improvements over state-of-the-art serving systems on GPUs. It enables efficient batching, handling dynamic sequence lengths and complex decoding algorithms, which is critical for scalable real-time inference. However, PagedAttention's optimizations primarily focus on multi-request serving workloads rather than single small model on-device inference.

Together, FlashAttention and PagedAttention address foundational and system-level memory challenges in Transformer inference. FlashAttention reduces overall memory IO at the kernel algorithmic level, while PagedAttention enhances memory utilization and scheduling efficiency in deployable serving systems.

## Post-Training Quantization Methods: AWQ and GPTQ

Quantization of weights to 4-bit representations has become a key enabler for deploying language models on constrained hardware without fine-tuning. AWQ (Activation-aware Weight Quantization) [4] and GPTQ (Generalized Post Training Quantization) [5] are leading post-training quantization algorithms that minimize accuracy loss while significantly reducing model size and inference latency.

AWQ identifies a small subset (~1%) of activation-salient weight channels that strongly influence model outputs and applies per-channel scaling factors computed via calibration data. This activation-aware scaling retains near-full precision accuracy with uniform 4-bit weight quantization and avoids mixed-precision overhead during inference. AWQ's calibration process is faster than GPTQ's and often achieves lower perplexity in instruction-tuned and multi-modal models. Deployment frameworks such as HuggingFace Transformers integrate AWQ with optimized execution kernels.

GPTQ quantizes weights layer-wise using an iterative approach guided by second-order Hessian information to minimize output error. It is mathematically analogous to Babai's nearest plane algorithm, offering a geometric interpretation of quantization accuracy. GPTQ achieves aggressive compression down to 3-4 bits per weight while maintaining accuracy in models up to 175B parameters. However, its calibration process is longer and can be prone to overfitting on small calibration sets.

Comparative evaluations [6][7] show AWQ and GPTQ yield roughly 4x reduction in memory footprint and comparable latency improvements, enabling deployment of small and moderately large LMs on GPUs such as NVIDIA A100 and H100. Both techniques rely on small calibration datasets and incur some preprocessing time, but simplify deployment by eliminating retraining. Application to small models like the Qwen2.5 demonstrates benefits in on-device inference and hardware acceleration [13].

## Speculative Decoding and Adaptive Sampling for Efficient Inference

Speculative decoding methods [8][9][10][11] accelerate inference by generating multiple draft tokens per step with a smaller draft model and verifying them in parallel with a larger target model. Recursive Speculative Decoding enhances diversity by sampling draft tokens without replacement and early truncation of unlikely sequences, improving efficiency without output quality loss.

SpecTr formulates speculative decoding as an optimal transport problem, enabling near-optimal acceptance probabilities and speedups up to 2.13x for large LLMs. Reinforcement learning-based adaptive sampling dynamically balances latency and accuracy during inference by deciding sample stopping criteria, optimizing compute resources in real time [14].

Industrial deployments at Google incorporate speculative decoding variants such as tree-based draft tokens and multi-head predictors to reduce latency with significant speed-ups of ~2x to 3x in production systems [10]. PyTorch frameworks have released practical implementations yielding up to 2x speedups with lightweight KV-cache management and attention mask modifications for correctness verification [11].

While most research targets large LLMs, the principles extend to small language models. However, explicit work on very small models like Phi-3 or Qwen small variants remains limited, suggesting valuable future research directions.

## Theoretical Advances in Composable Decoding Frameworks

Recent work proposes composable decoding on the probability simplex to unify multiple sampling strategies into a single optimization framework balancing expected model score and diversity regularization [12]. This framework supports the construction of novel decoders without requiring external rewards or model updates.

Such theoretical grounding offers future opportunities to design efficient decoding algorithms that can adaptively trade-off between sample quality and computational cost, an important consideration for small model inference where resources are constrained.

## Trends and open problems

Efficient inference for small language models continues to advance with integrated hardware-aware algorithms and system-level memory management. FlashAttention and vLLM PagedAttention form a complementary foundation by addressing GPU memory IO and serving efficiency, respectively. Adoption in both training and inference pipelines is growing.

Post-training quantization methods like AWQ and GPTQ provide practical and scalable solutions to reduce memory and latency overhead without retraining. Future work could further optimize calibration processes, reduce preprocessing latency, and extend methods to newer model families.

Speculative decoding and adaptive sampling algorithms achieve significant inference speedups by exploiting draft models and parallel verification, but require further adaptation and benchmarking on small models such as Phi-3 and Qwen small.

Theoretical compositional decoding frameworks suggest a promising direction for principled algorithm design that balances quality, diversity, and computation.

Open problems include:

- Benchmarking and optimizing kernel-level inference algorithms (FlashAttention, PagedAttention) specifically for very small models and diverse hardware.
- Developing quantization methods with faster calibration, better robustness, and support for extreme low bitwidths in on-device environments.
- Extending speculative decoding strategies with adaptive mechanisms tailored to small models' constraints and accuracy requirements.
- Enhancing theoretical decoding frameworks with practical implementations benefiting efficient small LM use cases.
- Comprehensive evaluations of integrated inference pipelines combining quantization, optimized attention kernels, and advanced decoding.

Continued research requires tight collaboration between algorithm design, system implementation, and hardware-aware optimization to fully realize efficient small language model inference at scale.

---

*Report compiled with evidence from [1][2][3][4][5][6][7][15][13][8][9][10][11][12][14].

## References
[1] FLASHATTENTION: Fast and Memory-Efficient Exact Attention with IO-Awareness. arxiv. https://arxiv.org/abs/2205.14135 (2022-06-23)
[2] Paged Attention - vLLM Documentation. web. https://docs.vllm.ai/en/latest/design/paged_attention/ (n.d.)
[3] Efficient Memory Management for Large Language Model Inference (PagedAttention and vLLM). arxiv. https://arxiv.org/abs/2309.06180 (n.d.)
[4] AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. hf-search. https://huggingface.co/papers/2306.00978 (2023-06-01)
[5] The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm. hf-search. https://huggingface.co/papers/2507.18553 (2025-07-24)
[6] Accelerating LLM inference with post-training weight and activation quantization using AWQ and GPTQ on Amazon SageMaker AI. web. https://aws.amazon.com/blogs/machine-learning/accelerating-llm-inference-with-post-training-weight-and-activation-using-awq-and-gptq-on-amazon-sagemaker-ai/ (2026-01-09)
[7] A practical guide to INT4 quantization for small language models: GPTQ vs AWQ, Olive, and real-world results. web. https://medium.com/data-science-at-microsoft/a-practical-guide-to-int4-quantization-for-slms-gptq-vs-awq-olive-and-real-world-results-2f63d6963d1d (2026-02-24)
[8] Recursive Speculative Decoding: Accelerating LLM Inference via Sampling Without Replacement. hf-search. https://huggingface.co/papers/2402.14160 (2024-02-21)
[9] SpecTr: Fast Speculative Decoding via Optimal Transport. hf-search. https://huggingface.co/papers/2310.15141 (2023-10-23)
[10] Looking back at speculative decoding - Google Research. web. https://research.google/blog/looking-back-at-speculative-decoding (2024-12-06)
[11] A Hitchhiker’s Guide to Speculative Decoding - PyTorch. web. https://pytorch.org/blog/hitchhikers-guide-speculative-decoding/ (2024-11-13)
[12] Composable Decoding on the Probability Simplex: Theory and Implementation. hf-search. https://huggingface.co/papers/2609.34992 (2026-09-28)
[13] On-Device Qwen2.5: Efficient LLM Inference with Model Compression and Hardware Acceleration. hf-search. https://huggingface.co/papers/2504.17376 (2025-04-24)
[14] Small RL Controller, Large Language Model: RL-Guided Adaptive Sampling for Test-Time Scaling. hf-search. https://huggingface.co/papers/2606.03102 (2026-06-02)
[15] A Comprehensive Evaluation of Quantized Instruction-Tuned Large Language Models. arxiv. https://arxiv.org/abs/2409.11055 (2024-09-10)
