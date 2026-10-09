# Survey on World Models: Core Methods and Recent Advances

## TL;DR
- Foundational world models combine latent space representations via variational autoencoders with recurrent neural networks (MDN-RNNs) to model stochastic environment dynamics enabling planning and reinforcement learning entirely in learned latent "dream" spaces [1][2][3][4].
- Recent advances like DreamerV3 and MuZero employ robust recurrent state-space models and planning in latent spaces, improving exploration, scalability, and policy performance across hundreds of complex domains without environment-specific tuning [5][6].
- The emergence of large-scale generative world models like Genie expands the concept to open-ended interactive environments trained on unlabeled video data, pushing world models towards generalist AI applications [7].
- Action-conditioned video world models leverage action-conditioned video generation architectures to tightly integrate perception and prediction, enabling more physically plausible, multi-view, and long-horizon embodied control scenarios [8][9][10][11][12].
- Challenges remain in hierarchical planning, balancing visual fidelity with predictive efficiency, dataset scale, and bridging simulated latent state representations with real-world control tasks.

## Background

World models stem from foundational work by Schmidhuber and colleagues beginning in the 1990s, which introduced the use of recurrent neural networks (RNNs) as predictive world models capable of supporting planning and reinforcement learning through supervised prediction of environment dynamics [4]. These early works incorporated the concept of artificial curiosity to motivate exploration and combined recurrent models with controllers capable of learning behaviors through prediction and planning.

A canonical architecture established later separates the world model into two modules: a V model compressing high-dimensional observations into low-dimensional latent embeddings via convolutional variational autoencoders (ConvVAEs), and an M model that models temporal dynamics in the latent space using recurrent neural networks combined with a mixture density network (MDN-RNN) to probabilistically predict distributions over future embeddings under actions [1][2][3].

This configuration enables the use of learned latent "dream" environments 
 simulated rollout trajectories in latent spaces  where controllers can be trained efficiently for decision making, transferring learned policies back to the real environment. Key benefits include compact representation, probabilistic modeling of uncertainty, and accelerated policy search [1][2][3].

## Foundational Core Technical Methods

The foundational works [1][2][3][4] emphasize three core technical components:

1. **Latent Space Encoding with VAEs:** Raw observations, often high-dimensional images, are compressed into latent vectors (z) by the V model, a convolutional variational autoencoder trained by minimizing reconstruction loss. This reduces computational cost and focuses modeling capacity on salient features [1]. However, some limitations arise as the latent space may encode irrelevant features, potentially requiring retraining or task-specific adaptation.

2. **Predictive Temporal Modeling with Recurrent MDN Networks:** The M model is typically an LSTM augmented with a mixture density output layer that predicts the probability distribution over next latent states given current latent encodings, actions, and hidden states. This stochastic approach captures environment uncertainty and complex temporal dependencies [1][2][3].

3. **Controller Training in Latent Simulated Environments:** Controllers receive current latent states and RNN hidden states as inputs and are trained via reinforcement learning or evolutionary strategies inside the learned world model, allowing efficient policy optimization without interaction with the real environment [1][2].

These technical elements create a framework where agents can train entirely within their dream environments, greatly accelerating learning. Early RNN world modeling work laid this conceptual and algorithmic foundation, but limitations include relatively simple non-hierarchical planning and challenges in encoding task-relevant latent features [4]. Extensions have proposed hierarchical models and unified architectures combining model and controller networks [2].

## Recent Advances in World Model Architectures

Recent literature demonstrates significant advances built on the foundational architecture that enhance generality, stability, and exploration:

### DreamerV3 and Variants

DreamerV3 [5] introduces improvements to stabilize training across diverse visual environments and task domains using a Recurrent State-Space Model (RSSM) with several normalization and balanced loss techniques (e.g., symlog transformations, free bits for KL loss clipping). It is trained jointly with a critic and an actor, allowing imagination-driven policy improvement.

DreamerV3 shows remarkable robustness, success in complex environments like Minecraft diamond collection from raw pixels without expert input, and scalability across 150+ benchmarks with fixed hyperparameters. Extensions such as DreamerV3-XP [13] enhance exploration through prioritized replay buffers and uncertainty estimation leading to faster learning.

### MuZero and Planning in Latent Spaces

MuZero [10] learns a latent dynamics model optimized for policy, value, and reward prediction without requiring explicit environment dynamics or pixel reconstruction, integrating model learning tightly with Monte-Carlo Tree Search (MCTS) for planning. It achieves superhuman results in games and establishes a foundation for combining learned latent models with search-based planning.

MuDreamer, a reconstruction-free variant inspired by DreamerV3 and MuZero, replaces pixel reconstruction with prediction of environment rewards and action-conditioned dynamics to improve robustness to visual distractions and training efficiency [2].

### Generative Interactive Environments (Genie)

Genie [7] represents a large-scale generative world model constructed from unlabeled video datasets using spatiotemporal tokenization and autoregressive modeling, with 11B parameters enabling agents to learn from previously unseen video behaviors. It introduces a new direction towards more open-ended, data-driven interactive virtual worlds beyond hand-designed simulators.

### Exploration Enhancements

Optimistic World Models [8] and extensions of DreamerV3 explore incorporating optimism biases in the learned models and uncertainty-based intrinsic rewards to encourage efficient exploration in challenging sparse-reward environments, demonstrating substantial empirical improvements in benchmarks like Atari and DeepMind control tasks.

## Technical Methods in Action-Conditioned Video World Models

The last several years have seen the emergence of video-conditioned world models that tightly integrate visual perception with action-conditioned future prediction. These models are integral to embodied environments such as robotics and procedural task execution [11][12].

### Core Architectures

Transformer-based video generation models such as VideoGPT [14] and diffusion models [15] form the backbone of many action-conditioned video world models. These models predict future frames conditioned on past observations and future actions, combining visual feature transformers and autoregressive or diffusion-based generation approaches.

Notable approaches like DreamTrue [11] and 0-WM [15] introduce multi-view and multi-modal conditioning, counterfactual post-training, and cross-embodiment action conditioning to improve physical plausibility and action-following fidelity.

UNITAS [16] pushes spatial reasoning further with 3D-native world action models integrating metric spatial coordinates and video latents, enabling more precise embodied manipulation modeling.

### Training Regimes and Data

Training methods utilize varied data sources including large-scale robot teleoperation trajectories, human egocentric videos, and teleoperated failure sequences, combining supervised learning of actions, video prediction, and recognized task progress signals [15]. Cross-view and multi-agent synchronization enhance modeling of interactions [13]. Some methods adopt purely offline training without real-world data, enabling zero-shot generalization [17].

### Challenges

Crucial challenges include bridging between predicted visual futures and executable robot actions, coping with sparse labeled action data, and computational efficiency for real-time control [18]. Studies highlight limitations in modeling complex physical interactions (ACWM-Phys [19]) and the need for scalable yet interpretable action-conditioned video prediction architectures.

## Trends and open problems

Several broad trends emerge from the synthesis of foundational and recent literature:

- **Latent Space Modeling and Stochastic Dynamics**: Latent variable models combined with probabilistic RNNs remain central, with ongoing work integrating more expressive models and hierarchical structures.

- **Generalization Across Diverse and Complex Domains**: Models like DreamerV3 and Genie demonstrate scaling world models to hundreds of tasks and open-world video datasets, but generalization beyond benchmark suites to real applications requires further study.

- **Exploration via Uncertainty and Optimism**: Incorporation of intrinsic motivation, uncertainty estimation, and optimism in model training improves exploration efficacy but raises questions on efficient scaling and proper balance.

- **Bridging World Models and Real-World Control**: Transferring from learned latent dream environments or video-conditioned predictions to physical robot control remains challenging due to calibration, embodiment differences, and sparse data.

- **Action-Conditioned Video Prediction Advances**: Multimodal transformers, diffusion models, and unified video-action models advance the fidelity and applicability of action-conditioned video world models, though computational cost and long-horizon state consistency require further innovation.

- **Hierarchical and Abstract Planning**: Foundational works and later extensions suggest the importance and difficulty of hierarchical planning beyond stepwise prediction; concrete architectures remain an active research frontier.

- **Data Scale and Heterogeneity**: Mobilizing large collections of unlabeled video, multi-agent, and multimodal data offers opportunities but also demands efficient training methods and better unsupervised representation learning.

- **Safety and Robustness in Model-Based RL**: Emerging works integrating safety constraints into world models like Safe DreamerV3 highlight the growing importance of safe and reliable deployment.

In conclusion, world models have evolved from foundational RNN latent-space predictors rooted in Schmidhuber's early work to diverse, scalable, and multimodal frameworks integrating robust training, planning, and video generation. Future progress will depend on overcoming hierarchical reasoning challenges, improving real-world transfer, and balancing scalable data-driven representation learning with computational efficiency and safety assurances.

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-03-27)
[2] Recurrent world models facilitate policy evolution. web. https://dl.acm.org/doi/10.5555/3327144.3327171 (2018-12-03)
[3] World Models Official Project Website. web. https://worldmodels.github.io/ (2018-03-27)
[4] 1990: Planning & Reinforcement Learning with Recurrent World Models and Artificial Curiosity. web. https://people.idsia.ch/~juergen/world-models-planning-curiosity-fki-1990.html (n.d.)
[5] Mastering Diverse Domains through World Models (DreamerV3). hf-search. https://huggingface.co/papers/2301.04104 (2023-01-10)
[6] Policy-shaped prediction: avoiding distractions in model-based reinforcement learning. hf-search. https://huggingface.co/papers/2412.05766 (2024-12-08)
[7] Genie: Generative Interactive Environments. hf-search. https://huggingface.co/papers/2402.15391 (2024-02-23)
[8] Optimistic World Models: Efficient Exploration in Model-Based RL. web. https://arxiv.org/html/2602.10044v1 (2026-02-17)
[9] DreamerV3-XP: Optimizing Exploration through Uncertainty Estimation. web. https://arxiv.org/html/2510.21418v1 (2025-10-21)
[10] MuZero: Mastering Games by Planning with a Learned Model. hf-search. https://huggingface.co/papers/1911.08265 (2019-11-19)
[11] DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training. arxiv. https://arxiv.org/abs/2610.12468 (2026-10-08)
[12] WorldGuide: Goal-Directed Video World Model for Procedural Task Execution. arxiv. https://arxiv.org/abs/2610.12459 (2026-10-08)
[13] Multi-Agent Egocentric World Model with Fine-Grained Embodied Interaction. arxiv. https://arxiv.org/abs/2610.12299 (2026-10-08)
[14] iVideoGPT: Interactive VideoGPTs are Scalable World Models. hf-search. https://huggingface.co/papers/2405.15223 (2024-05-24)
[15] 0-WM: A Unified Video-Action World Model for Robotic Manipulation. web. https://arxiv.org/html/2606.01027 (n.d.)
[16] UNITAS: A 3D-Native World Action Model for Embodied Manipulation. arxiv. https://arxiv.org/abs/2610.12099 (2026-10-08)
[17] Aether: Geometric-Aware Unified World Modeling. hf-search. https://huggingface.co/papers/2503.18945 (2025-03-24)
[18] WAM-Cache: Staleness-Bounded KV Reuse for Efficient World Action Models. arxiv. https://arxiv.org/abs/2610.11401 (2026-10-08)
[19] ACWM-Phys: Investigating Generalized Physical Interaction in Action-Conditioned Video World Models. hf-search. https://huggingface.co/papers/2605.08567 (2026-05-09)
