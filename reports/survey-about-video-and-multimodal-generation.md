# Survey on Video and Multimodal Generation

## TL;DR
- Modern video and multimodal generation methods are fundamentally rooted in Denoising Diffusion Probabilistic Models (DDPM) extended through latent diffusion and transformer-based architectures, facilitating scalable and high-fidelity generation [1][2][3][4].
- Recent advances in video diffusion leverage Diffusion Transformer (DiT) backbones exemplified by Sora and Movie Gen, which employ spatiotemporal attention and large-scale foundation models to achieve improved temporal coherence and multimodal conditioning, while challenges remain in long-range physical realism and video consistency [5][6][7][8][9].
- Multimodal generation beyond video integrates latent diffusion frameworks with specialized tokenizers and unified models such as KVAE and MUNI, enabling coherent generation across audio, image, and text modalities through joint latent spaces and autoregressive diffusion [10][11][12][13][14].
- Evaluation benchmarks like VBench and Video-Bench advocate multi-dimensional metrics spanning temporal consistency, semantic alignment, and human perceptual quality, employing a mix of classical perceptual scores and large language model based human-alignment scoring to capture the complex demands of multimodal video generation evaluation [15][16][17].

## Background
Diffusion models have emerged as a powerful generative paradigm by framing data synthesis as an iterative denoising process, a foundational principle described in Denoising Diffusion Probabilistic Models (DDPM) [1]. This foundational framework iteratively learns to reverse a gradual noising process, enabling high-quality sample generation that scales to images and beyond. Subsequently, latent diffusion methods optimize efficiency by conducting the diffusion process in a compressed latent space rather than pixel space, leveraging pretrained variational autoencoders (VAE) or similar encoders to map data to latent representations [2]. This reduces computational cost while retaining semantic fidelity, allowing extension to high-dimensional domains like video and multimodal data.

To handle the complexity of spatiotemporal video data, transformer-based architectures employing attention mechanisms along spatial and temporal dimensions have been incorporated, as seen in recent video diffusion models [1][3]. These models tokenize video and multimodal conditions into sequences processed jointly through transformer blocks, enabling fusion of heterogeneous signals such as text, camera angles, and depth for precise control. Approaches like FullDiT and OmniVDiff illustrate the use of unified transformer backbones incorporating adaptive conditioning strategies and modality-specific projection heads to handle multiple video modalities and conditions jointly [3][4]. This architectural evolution represents the current foundation upon which state-of-the-art video and multimodal diffusion models are built.

## Foundational Methods: Diffusion and Latent Representations
The seminal DDPM framework [1] formulates data generation as a learned iterative denoising process, reversing a fixed Gaussian noise addition schedule. This mechanism enables flexible generation of richly structured data. Latent diffusion models [2] build on this by operating diffusion processes within semantic latent spaces compressed by pretrained VAEs or tokenizers, markedly improving computational efficiency particularly for large video frames or multimodal inputs.

FullDiT [3] presents a unified transformer architecture incorporating multimodal control by tokenizing multiple conditioning inputs into concatenated sequences processed with full self-attention. It employs sophisticated positional encodings in 2D and 3D to capture spatial and temporal structure, alongside mechanisms like AdaLN-Zero for diffusion timestep conditioning inside attention layers. Similarly, OmniVDiff [4] extends the diffusion framework to jointly generate and understand multiple video modalities (e.g., RGB, depth, segmentation) by utilizing modality-specific projection heads and an adaptive generation/conditioning control per modality. Both leverage pretrained 3D causal VAEs to encode inputs into latent patches for efficient joint denoising.

While foundational methods reveal the power of diffusion combined with transformer architectures in latent spaces, challenges remain in computational overhead and dynamic adaptability across diverse modalities, motivating ongoing research in improved latent encodings and conditioning strategies.

## Advances in Video Diffusion Architectures
Recent research in video diffusion models highlights the development of novel transformer-based architectures that enhance generation fidelity, temporal consistency, and multimodal controllability 
[5][6][7][8][9]. The foundational architectures use the Diffusion Transformer (DiT) backbone, which tokenizes videos into spacetime patch tokens for subsequent transformer processing, enabling efficient spatiotemporal attention.

OpenAI's Sora model exemplifies these principles, employing a diffusion transformer trained on compressed latent representations of variable-resolution videos to generate high-fidelity outputs up to one minute with flexible dimensions [8]. Despite advances, Sora's physical realism and extremely long-duration video coherence remain challenging, highlighting ongoing open problems.

Beyond Sora, models like Movie Gen represent large-scale multimodal foundation models, supporting text-to-video generation, precise editing, video-to-audio generation, and video personalization through a 30 billion parameter Transformer architecture trained with Flow Matching objectives [7]. These models achieve state-of-the-art quality at 1080p resolution for up to 16-second video clips.

Other architectures, including Open-Sora and WorldGPT, combine spatial-temporal diffusion transformers with 3D autoencoders or multimodal learning integrated with language models to produce flexible, temporally consistent videos with rich conditioning capabilities [5][6]. Collective evidence illustrates a vibrant trajectory toward foundation-model-scale video diffusion transformers incorporating multiple conditioning modalities, improved training objectives, and scalable architectures.

## Core Methods in Multimodal Generation beyond Video
Multimodal generation pushes beyond single-video domains by integrating various modalities including audio, text, images, and video within unified generative frameworks. This field leverages latent diffusion methods extended with specialized tokenizers and multimodal architectures to achieve coherent synthesis across content types [10][11][12][13][14].

KVAE [10] introduces modality-specific tokenizers designed for audio, video, and image signals, essential for encoding input into compressed latent representations that support efficient diffusion process learning. Objective metrics like PSNR, LPIPS, and PESQ demonstrate their effectiveness.

The MUNI framework [11] builds a unified latent diffusion model that facilitates any-to-any generation tasks by jointly training modality-specific encoders and a shared flow-based latent prior, optimized with a routed training objective to enforce latent coherence and predictive sufficiency across modalities.

LatentLM [12] integrates autoregressive next-token diffusion with VAEs for continuous and discrete data modalities, enabling seamless multimodal generation and understanding within large language models. This approach improves image generation scalability and has demonstrated success in text-to-speech synthesis.

Cross-modal conditioning methods such as AudioToken [13], which adapts text-to-image diffusion models to generate images conditioned on audio inputs, and diffusion latent aligners [14] further enable joint and cross-modal content generation by synchronizing latent representations across modalities.

While promising, this domain requires robust benchmarks and training procedures to manage coherent minimal latent representations, as well as scalability to large diverse real-world datasets.

## Evaluation Techniques and Benchmarks
Accurate and comprehensive evaluation of video and multimodal generative models calls for multifaceted metrics that capture temporal, semantic, stylistic, and perceptual dimensions beyond traditional single-score indicators [15][16][17].

The VBench benchmark [15] exemplifies a comprehensive metric suite that disaggregates video generation quality into multiple dimensions, spanning temporal aspects like subject and background consistency, flickering, motion smoothness, and dynamic degree, as well as frame-wise qualities including aesthetic and imaging quality.

Disentangling video-condition consistency into semantic and stylistic factors, VBench measures object classes, human actions, spatial relationships, and scene attributes alongside appearance and temporal style using diverse metrics such as DINO and CLIP features, RAFT for motion smoothness, and LAION and MUSIQ for aesthetic and imaging quality.

The Video-Bench [16][17] framework adopts a two-fold approach categorizing metrics into traditional (IS, FID, FVD), embedding-based, and large language model-based human-aligned evaluations assessing video quality and condition consistency relative to text prompts. It additionally incorporates human annotations and supports large datasets like Kinetics-400 for dynamic action analysis.

Contemporary benchmark development converges on hybrid evaluation methodologies combining automated perceptual measures with language model based alignment scoring to approximate human judgment accurately. However, the complexity of video and multimodal generation necessitates multi-dimensional, disaggregated evaluation protocols that are challenging to scale fully without extensive human involvement.

## Trends and open problems
Video and multimodal generation research is rapidly evolving toward unified, large-scale foundation models that integrate multimodal conditioning and scalable architectures based on diffusion transformers and latent spaces. Core trends emphasize transformer-based spatiotemporal modeling for high-fidelity video generation, latent diffusion for efficiency, and multimodal tokenization for coherent cross-domain synthesis.

A key open problem is maintaining long-range temporal coherence and physical realism, especially for extended video durations, which current state-of-the-art models like Sora and Movie Gen only partially address [5][7][8][9]. Further, efficient adaptive conditioning techniques across heterogeneous modalities without bloating parameter counts remain vital.

The multimodal generation domain demands unified, scalable models addressing latent space coherence and cross-modal alignment with flexible conditioning and improved autoregressive diffusion techniques. Training on larger, more diverse multimodal datasets combined with better benchmarks could enhance model generality and real-world applicability.

Evaluation protocols require refinement to effectively measure multimodal generations on semantic, stylistic, temporal, and perceptual axes, integrating automated and human-aligned metrics harmoniously. The high annotation and computation cost remains a bottleneck, suggesting opportunities for learning-based or self-supervised evaluation methods.

Finally, open research toward models capable of interactive editing, personalization, and precise control of generated multimodal content while ensuring coherence and quality poses an exciting challenge that bridges core technical innovation and application-driven demands.

---

(Word count: approximately 1840 words)

## References
[1] Denoising Diffusion Probabilistic Models. arxiv. https://arxiv.org/abs/2006.11239 (2020-12-16)
[2] High-Resolution Image Synthesis with Latent Diffusion Models. arxiv. https://arxiv.org/abs/2112.10752 (2022-04-13)
[3] FullDiT: Video Generative Foundation Models with Multimodal Control. web. https://openaccess.thecvf.com/content/ICCV2025/papers/Ju_FullDiT_Video_Generative_Foundation_Models_with_Multimodal_Control_via_Full_ICCV_2025_paper.pdf (n.d.)
[4] OmniVDiff: Omni Controllable Video Diffusion for Generation and Understanding. web. https://ojs.aaai.org/index.php/AAAI/article/download/38068/42030 (n.d.)
[5] Open-Sora: Democratizing Efficient Video Production for All. hf-search. https://huggingface.co/papers/2412.20404 (2024-12-29)
[6] WorldGPT: A Sora-Inspired Video AI Agent as Rich World Models from Text and Image Inputs. hf-search. https://huggingface.co/papers/2403.07944 (2024-03-10)
[7] Movie Gen: A Cast of Media Foundation Models. web. https://arxiv.org/html/2410.13720v1 (n.d.)
[8] Video generation models as world simulators | OpenAI. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[9] Sora as a World Model? A Complete Survey on Text-to-Video Generation. web. https://arxiv.org/html/2403.05131v3 (n.d.)
[10] KVAE: Family of Tokenizers for Multimodal Generative Models. arxiv. https://arxiv.org/abs/2608.05798 (2026-08-06)
[11] MUNI: Multimodal Unified Latent Diffusion for Coherent Any-to-Any Generation. arxiv. https://arxiv.org/abs/2606.16408 (2026-06-15)
[12] Multimodal Latent Language Modeling with Next-Token Diffusion. hf-search. https://huggingface.co/papers/2412.08635 (2024-12-11)
[13] AudioToken: Adaptation of Text-Conditioned Diffusion Models for Audio-to-Image Generation. hf-search. https://huggingface.co/papers/2305.13050 (2023-05-22)
[14] Seeing and Hearing: Open-domain Visual-Audio Generation with Diffusion Latent Aligners. hf-search. https://huggingface.co/papers/2402.17723 (2024-02-27)
[15] VBench: Comprehensive Benchmark Suite for Video Generative Models. web. https://openaccess.thecvf.com/content/CVPR2024/papers/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.pdf (2024-04-27)
[16] Video-Bench: Human-Aligned Video Generation Benchmark. web. https://openaccess.thecvf.com/content/CVPR2025/papers/Han_Video-Bench_Human-Aligned_Video_Generation_Benchmark_CVPR_2025_paper.pdf (2025-06-01)
[17] Video-Bench GitHub Repository. web. https://github.com/Video-Bench/Video-Bench (n.d.)
