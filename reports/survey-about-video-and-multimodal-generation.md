# Survey on Video and Multimodal Generation: Core Technical Methods and Innovations

## TL;DR

- Denoising Diffusion Probabilistic Models (DDPM) laid the foundational framework for high-quality image and video generation, extended to videos by Video Diffusion Models (VDM) using factorized space-time attention and 3D U-Net architectures [1][2].
- Latent Diffusion Models (LDMs) significantly improved efficiency by performing diffusion in compressed latent spaces, allowing scalable, high-resolution video synthesis, forming the basis for several key video generation models [3].
- Diffusion Transformers (DiT) improve scalability and conditioning by applying transformers to latent patch tokens, providing a strong architecture component for text-to-video models like Sora that employ diffusion transformers to generate videos in compressed latent spaces [4][5][6].
- Multimodal generation leverages unified diffusion transformers operating on shared latent spaces integrating video, audio, text, and images, exemplified by UniForm and C3Net architectures enabling flexible conditional and joint multimodal synthesis [7][8].
- The Movie Gen model suite represents a recent state-of-the-art foundation for generative video and audio, achieving high-fidelity 1080p videos with synchronized audio, and supporting editing and personalization, trained on vast multimodal datasets [9].
- Architectural innovations focus on scalability and efficiency, as seen in Chimera and VDT, combining efficient attention mechanisms, mask modeling, and dual-stream processing to improve temporal coherence and generation quality across modalities [10][11][12].

## Background

Generative modeling of video and multimodal content has become pivotal in AI research, with underlying progress heavily influenced by the diffusion probabilistic model framework introduced by DDPMs. DDPMs facilitate high-fidelity iterative denoising sampling, originally applied to images but foundational to video models. Extensions such as Video Diffusion Models (VDM) factorize attention mechanisms spatially and temporally to address the challenges of modeling dynamic video content across frames [1][2].

Latent Diffusion Models (LDM), introduced circa 2021, revolutionized efficiency by compressing high-dimensional data into compact latent spaces where diffusion is performed, making scalable high-resolution video generation computationally feasible [3]. These latent models typically rely on encoder-decoder architectures based on variational autoencoders (VAEs) to transform between pixel and latent representations.

More recent architectures replace convolutional UNet backbones in diffusion models with transformer layers operating on discrete latent patch tokens, termed Diffusion Transformers (DiT), further improving model scalability, conditioning flexibility, and generation fidelity [4]. Large-scale diffusion transformers like Sora scale this paradigm to long, high-fidelity video generation within compressed latent patch spaces [5][6].

With the growing demand for content integrating multiple modalities such as video, audio, text, and images, multimodal diffusion transformers have been developed. These models commonly leverage unified latent spaces and transformers that jointly model or condition across modalities, supporting tasks like video-to-audio, text-to-video, and audio-video generation with tight temporal and semantic alignment [7][8].

Foundational and recent advances in movie and long-form video generation have led to the development of foundation models like Movie Gen, which are trained on massive datasets comprising hundreds of millions of videos and images, supporting high-fidelity 1080p video and cinematic audio generation with editing capabilities [9].

## Core Foundations of Video Diffusion Models

The Denoising Diffusion Probabilistic Model (DDPM) by Ho et al. has been seminal, inspiring adaptations to video generation by incorporating temporal dimensions explicitly [1]. The Video Diffusion Model (3D U-Net architecture) introduces factorized space-time attention to apprehend both spatial and temporal dependencies, enabling coherent video synthesis over a sequence of frames [1]. VDMs can condition on image frames or other labels for controllable video generation, showing efficacy in moderate-length video synthesis.

Latent Diffusion Models (LDMs) enhance this paradigm by encoding high-dimensional pixel data into latent representations via VAEs, performing diffusion in this compressed space to vastly reduce computational cost and enable high-resolution generation [3]. This method underlies several key video generation frameworks, including Sora.

Diffusion Transformer (DiT) architectures further refine this by swapping convolution-based UNet backbones for transformer backbones taking sequences of latent patch tokens. DiT uses adaptive layer normalization and residual scaling for conditioning, proving crucial for scalability and improved synthesis quality [4]. This design is employed in advanced models like Sora, which represents a leading text-to-video diffusion transformer architecture capable of generating minute-long videos efficiently [5][6].

However, challenges remain in generating very long videos with high fidelity, preserving multi-subject consistency, and managing computational costs [2].

## Multimodal Generation Techniques Combining Video and Additional Modalities

Recent work has emphasized generating multimodal data with shared latent spaces and diffusion transformers that handle video alongside audio, text, and images, enabling flexible and controllable modalities.

UniForm presents a unified diffusion transformer generating audio and video simultaneously by concatenating latent features from audio and video VAEs into a shared latent space. This model supports multiple generation tasks, including video-to-audio and text-to-audio-video synthesis, without requiring task-specific fine-tuning, achieving state-of-the-art generation quality and multimodal alignment [7].

C3Net aligns multimodal conditions such as audio, text, and images into a shared semantic latent space using modality-specific encoders trained with contrastive objectives. It uses a trainable Control C3-UNet for compound conditional generation, enabling flexible and improved multimodal synthesis quality over linear interpolation approaches [8].

Other models like BLIP3-o extend multimodal architectures integrating text, image, video, and audio modalities for unified generation and understanding, employing diffusion transformers within a single framework [13]. VGDFR and Latte optimize video generation efficiency and quality via dynamic latent frame rate adaptation and latent diffusion transformers operating in compressed latent spaces [14][15].

Such models highlight ongoing efforts to achieve cohesive multimodal generation, yet integrating video fully with other modalities in one model remains an open problem due to dataset scale and modality heterogeneity challenges.

## Architectural and Training Innovations in Video and Multimodal Diffusion Models

Architectural advances focus on improving scalability, efficiency, and temporal coherence in diffusion transformer-based video and multimodal generation.

Chimera proposes a hybrid diffusion transformer backbone combining kernel-based Kimi Delta Attention (KDA) with linear complexity for long-context tracking, Multi-head Latent Attention (MLA) for global context, and modality-aware convolutions for local spatiotemporal features. This design mitigates quadratic attention costs enabling large-scale, high-resolution visual generation including long videos and multimodal data [11].

VDT introduces mask modeling with spatial-temporal attention in transformer architectures, providing general-purpose video diffusion capabilities handling diverse video generation tasks with improved quality and generality [10].

Multimodal models such as LetsTalk fuse image, video, and audio modalities using modular attention mechanisms in a latent diffusion transformer framework to deliver realistic, temporally coherent talking video synthesis [16]. ProAV-DiT improves efficiency and synchronization by projecting audio and video into a shared latent space and employing dual-stream diffusion transformers for aligned cross-modal generation [12].

HunyuanVideo-Foley models aligned video-to-audio synthesis leveraging scalable data pipelines, self-supervised audio features, and dual-stream fusion with cross-attention mechanisms, representing advances in high-fidelity multimodal diffusion synthesis [17].

Despite these advances, benchmark evaluations and ablation studies remain sparse, limiting detailed understanding of trade-offs between architectural designs and training strategies.

## Foundations and Recent Advances in Movie and Long-Form Video Generation

Movie Gen by Meta represents a pioneering foundation model suite for video and audio generation, incorporating 30B and 13B parameter transformers for 1080p HD video and cinematic audio synthesis, trained on around 100 million videos and 1 billion images [9]. This large-scale pretrained model learns diverse phenomena including object motion, camera movement, and physics, supporting applications from text-to-video synthesis to video personalization and editing.

Captain Cinema introduces text-driven short movie generation using a top-down keyframe planning approach coupled with bottom-up multimodal diffusion transformers to ensure temporal coherence and content quality [18]. CineAGI employs multi-agent orchestration powered by large language models to coordinate narrative elements, maintain character consistency, and synchronize audio-visual components in long sequences [19].

Complementary research focuses on movie narration and understanding: Cue2Narrate proposes a pipeline jointly predicting what and when to narrate in extended clips, addressing narration generation in cinematic contexts [20]. Retrieval-Augmented Generation tackles challenges in multimodal story-level reasoning using selective retrieval from videos to enhance grounded generative models [21].

This area is very recent with foundational works mostly from 2024 onward, emphasizing growing interest and rapid innovation in multimodal movie generation and understanding.

## Trends and open problems

Despite substantial advancements, several challenges remain:

- **Temporal Coherence and Multi-subject Consistency:** Scaling generation length while maintaining coherent object and subject consistency over time continues to challenge video diffusion models [2].
- **Multimodal Integration:** Full seamless conditioning and joint generation across multiple modalities, especially integrating long videos with synchronized audio, text, and images in a single model, is an actively researched problem [7][8].
- **Scalability and Efficiency:** Quadratic attention complexity limits transformer scaling for long videos; innovations like Chimera’s kernel attention and mask modeling in VDT address this but require further work to generalize and standardize [10][11].
- **Benchmarking and Evaluation:** There is a lack of detailed benchmarking and ablation studies that uniformly evaluate architectures and training paradigms, impeding systematic assessment of gains and trade-offs [2][17].
- **Data Scale and Quality:** Large-scale multimodal datasets are essential for training foundation models but acquiring and curating such datasets remain costly and complex, impacting model robustness [9].
- **Editing and Personalization:** While models like Movie Gen introduce editing and personalization post-training, developing efficient, precise, and user-friendly mechanisms remains an open frontier [9].
- **Narrative Consistency in Long Form Video:** Models for long-form movie generation need to better capture narrative elements, character consistency, and story coherence beyond frame-level generation [18][19].

Future efforts will likely focus on addressing these challenges through architectural innovation, training strategies, large-scale diverse datasets, and improved evaluation metrics.

---

This survey has synthesized foundations and recent state-of-the-art research across video and multimodal generation leveraging diffusion and transformer-based models. It highlights the rapid progress and challenges lying ahead for generating coherent, high-fidelity, and multimodal video content with broad applications in multimedia AI.

## References
[1] Video Diffusion Models (Ho et al. 2022). arxiv. https://arxiv.org/abs/2204.03458 (2022-04-07)
[2] Video diffusion generation: comprehensive review and open challenges. web. https://link.springer.com/article/10.1007/s10462-025-11331-6 (2025-08-20)
[3] High-resolution image synthesis with latent diffusion models. arxiv. https://arxiv.org/abs/2112.10752 (2021-12-20)
[4] Diffusion Transformers for Image Generation (DiT). arxiv. https://arxiv.org/abs/2301.12864 (2023-01-30)
[5] Sora as a World Model? A Complete Survey on Text-to-Video Generation. arxiv. https://arxiv.org/abs/2403.05131 (n.d.)
[6] Video generation models as world simulators | OpenAI. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[7] UniForm: A Unified Diffusion Transformer for Audio-Video Generation. hf-search. https://huggingface.co/papers/2502.03897 (2025-02-06)
[8] C3Net: Compound Conditioned ControlNet for Multimodal Content Generation. web. https://openaccess.thecvf.com/content/CVPR2024/papers/Zhang_C3Net_Compound_Conditioned_ControlNet_for_Multimodal_Content_Generation_CVPR_2024_paper.pdf (N/A)
[9] Movie Gen: A Cast of Media Foundation Models. arxiv. https://arxiv.org/abs/2410.13720 (2024-10-04)
[10] VDT: General-purpose Video Diffusion Transformers via Mask Modeling. hf-search. https://huggingface.co/papers/2305.13311 (2023-05-22)
[11] Chimera: Designing and Chinchilla-Scaling Hybrid Visual Diffusion Transformers. arxiv. https://arxiv.org/abs/2607.28611 (2026-07-30)
[12] ProAV-DiT: A Projected Latent Diffusion Transformer for Efficient Synchronized Audio-Video Generation. hf-search. https://huggingface.co/papers/2511.12072 (2025-11-15)
[13] BLIP3-o: A Family of Fully Open Unified Multimodal Models-Architecture, Training and Dataset. hf-search. https://huggingface.co/papers/2505.09568 (2025-05-14)
[14] VGDFR: Diffusion-based Video Generation with Dynamic Latent Frame Rate. hf-search. https://huggingface.co/papers/2504.12259 (2025-04-16)
[15] Latte: Latent Diffusion Transformer for Video Generation. hf-search. https://huggingface.co/papers/2401.03048 (2024-01-05)
[16] LetsTalk: Latent Diffusion Transformer for Talking Video Synthesis. hf-search. https://huggingface.co/papers/2411.16748 (2024-11-24)
[17] HunyuanVideo-Foley: Multimodal Diffusion with Representation Alignment for High-Fidelity Foley Audio Generation. hf-search. https://huggingface.co/papers/2508.16930 (2025-08-23)
[18] Captain Cinema: Towards Short Movie Generation. hf-search. https://huggingface.co/papers/2507.18634 (2025-07-24)
[19] CineAGI: Character-Consistent Movie Creation through LLM-Orchestrated Multi-Modal Generation and Cross-Scene Integration. hf-search. https://huggingface.co/papers/2604.23579 (2026-04-26)
[20] From Visual Cues to Spoken Narration: Rethinking Audio Description. arxiv. https://arxiv.org/abs/2609.01725 (2026-09-01)
[21] Beyond Visual Boundaries: Rethinking Scene Segmentation for Movie RAG. arxiv. https://arxiv.org/abs/2608.28699 (2026-08-27)
