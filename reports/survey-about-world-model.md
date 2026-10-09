# Deep Survey on World Models in Machine Learning

## TL;DR

- Foundational world models by Ha and Schmidhuber (2018) establish a compact, unsupervised learning approach using recurrent neural networks with mixture density outputs to capture spatiotemporal environment representations, coupled with compact controllers trained by evolutionary strategies [1][2].
- Recent advancements exemplified by DreamerV3 extend these models with extensive normalization and robustness techniques, scalable architectures, and outperform MuZero on diverse tasks including Atari with fewer resources [3].
- Genie advances large-scale unsupervised generative world models based on video tokenization and transformer architectures, enabling interactive agent training in diverse, unlabeled scenes but faces challenges with precise physical modeling [4].
- MuZero remains a landmark method integrating learned models with Monte Carlo Tree Search planning, achieving superhuman results in discrete-action games, differing from earlier recurrent model architectures by using implicit planning models [5].
- Action-conditioned video world models constitute a specialized branch focusing on video prediction conditioned on actions to enable control in robotics and interactive environments, using geometric attention and unified policy-prediction frameworks but face challenges in real-time responsiveness and sim-to-real transfer [6][7][8][9].


## Background

World models refer to internal predictive models that agents learn to represent and simulate the dynamics of their environment. The seminal work by Ha and Schmidhuber (2018) framed this in the context of deep learning by training a generative recurrent neural network to compress spatiotemporal environment data, equipped with a controller network trained to act within the learned world model [1]. They employed LSTM architectures combined with mixture density networks to model observation distributions and trained with backpropagation, while using evolutionary strategies to train the policy controller [1][2]. This approach enabled agents to learn compact latent dynamics and plan policies by interacting mostly within a hallucinated simulated environment created by the world model.

The contribution of Ha and Schmidhuber laid a practical foundation drawing from decades of RNN and model-based reinforcement learning work. It demonstrated that unsupervised predictive modeling combined with compact policy evolution can solve complex tasks efficiently [1][2]. However, these foundational models primarily used recurrent architectures limited by memory constraints over long sequences and relatively small-scale environmental complexity.

Contrastively, more recent developments have pursued scaling, robustness, and generalization. This includes the evolution of architectures incorporating normalization techniques, stronger optimizers, and big-data training methods. Parallel lines in model-based planning and generative video models have pushed capabilities in both simulated and real-world-like environments.


## Foundational World Models: RNN-Based Learning and Policy Evolution

The earliest impactful approach to world models considered was the combination of recurrent neural networks to learn environment state predictions and evolutionary strategies to train smaller controllers operating within the learned latent space [1][2]. The RNN architecture, notably using LSTM cells, served to capture temporal dependencies and generate probabilistic predictions via mixture density outputs to represent uncertainty. This unsupervised learning setting was essential to creating a general-purpose, adaptive world model without requiring explicit environment labels or known dynamics.

A separate, smaller neural controller was trained to map the learned internal representation features into actions optimizing task performance. Training this policy with evolutionary strategies, instead of backpropagation, helped simplify credit assignment due to the complexity of differentiating through long time horizons in the latent space. This division allowed fast and compact policy optimization distinct from the large predictive model training.

Experimental validations in OpenAI Gym environments demonstrated that policies trained entirely within the hallucinated world model were transferable back to their real environments, validating the practical utility of the approach [2]. This represented a technical step forward in unsupervised model-based reinforcement learning. However, reliance on RNNs and MDN meant limitations in long-term memory capacity and difficulty scaling to more complex or high-dimensional environments, motivating further innovations.


## Recent Advances: Architectural Robustness, Scale, and Generality in DreamerV3 and Genie

DreamerV3 represents a significant leap forward in the evolution of world models leveraging recurrent state-space architectures combined with extensive architectural and training robustness innovations [3]. It introduces components such as block gated recurrent units (block GRU), RMS normalization, sigmoid linear unit (SiLU) activations, and advanced optimizers like LaProp. DreamerV3 also employs data balancing techniques—such as percentile return normalization, Kullback-Leibler divergence balancing, and free bits—that stabilize learning and significantly improve training robustness across diverse tasks [3].

Benchmarks show DreamerV3 surpassing state-of-the-art prior models including MuZero in Atari game performance while using fewer computational resources and fixed hyperparameters, signaling stronger generality and sample efficiency [3][5]. It notably scales to over 150 tasks including continuous and discrete action spaces, and 2D and 3D domains, demonstrating wide applicability.

Genie offers a starkly different vector in the world model research landscape by focusing on large-scale unsupervised generative video models trained on unlabeled internet videos [4]. It incorporates a massive 11 billion parameter transformer-based architecture unifying video frame tokenization and autoregressive dynamics modeling. Such scale enables long-range spatiotemporal modeling, supporting agent interaction via action-conditioned generation even on unseen scenes, which is a leap toward generalist agent training [4].

Despite Genie’s impressive scale and unsupervised learning prowess, it faces challenges modeling precise physical interactions and object dynamics, areas wherein DreamerV3 and MuZero maintain an edge through their more explicit latent state and planning formulations [3][4][5]. These complementary approaches highlight a research frontier prioritizing both scalable knowledge capture and physical fidelity.


## MuZero and the Evolution of Implicit Planning in Learned Models

MuZero introduced a distinct model-based reinforcement learning paradigm by learning implicit environment dynamics including policy, value, and reward functions optimized for planning within a learned latent space [5]. It combines this with Monte Carlo Tree Search (MCTS) to enable planning in complex discrete-action domains such as board games and Atari, achieving superhuman performance [5]. Unlike the explicit recurrent world models of the foundational era, MuZero’s model is specialized toward planning and evaluation functions enabling deeper lookahead without known environment rules.

Training is based on self-play, using the learned model predictions to guide Monte Carlo planning, differing from the Ha and Schmidhuber approach that evolves policies within hallucinated model rollouts [1][5]. MuZero exemplifies an effective fusion of model-based RL with classical planning methods, yet its discrete-action planning focus contrasts with the more general continuous control applicability of DreamerV3.

Subsequent extensions have aimed to improve MuZero’s applicability to large or continuous action spaces but challenges remain comparatively in scalability and ease of training [5]. Furthermore, the architectural complexity of MCTS planning in latent spaces introduces implementation difficulties relative to the more end-to-end learned architectures in recent world models [3][5].


## Action-Conditioned Video World Models: Integration of Video Prediction and Control

Recent work on action-conditioned video world models embodies a specialized trajectory in world models blending video prediction with control conditioning to enable tightly integrated perception and action systems in robotics and interactive environments [6][7][8][9]. These models commonly represent environments through sequences of images or frames conditioned explicitly on agent actions, aiming to faithfully predict future observations aligned to commanded behaviors.

Top methods emphasize mechanisms for encoding geometric transformations and preserving scene consistency, such as using SE(3) attention mechanisms to model rigid-body motions and multi-view observations integration [7]. Architectures often employ cross-attention to enhance conditioning on high-dimensional actions and unify policy learning with video generation, adopting diffusion models or autoregressive frameworks for sequential predictions [6][8].

Benchmarks demonstrate strong performance on various robotic manipulation datasets and competitions, yet challenges persist including faithful rollout adherence to actions, maintaining temporal consistency over long horizons, and bridging the gap to real-world physical interaction through sim-to-real transfer [6][7][9].

Surveys identify key research gaps such as incorporating tactile and multimodal sensing beyond vision, improving uncertainty estimation, and enhancing real-time interactive responsiveness and memory capabilities [9]. The computational costs and appearance biases inherent to video-based models also constrain deployment in time-critical systems.


## Trends and open problems

The survey of foundational and modern world models reveals clear research trends towards: (1) scaling architectures to improve generality and robustness, as seen in DreamerV3 and Genie; (2) combining latent state modeling with explicit planning, typified by MuZero; and (3) integrating video prediction with control for embodied interaction in action-conditioned models.

A major open challenge is the realistic modeling of physical interaction dynamics at scale, where unsupervised generative video models like Genie excel in scale and diversity but struggle with precise physics, while models like DreamerV3 and MuZero can address physics in narrower domains but less so in open-ended settings. Bridging sim-to-real gaps, enhancing tactile sensing and multimodal inputs, and improving long-horizon planning and memory are critical milestones.

Another frontier includes reconciling the planning complexity and discrete action focus of MuZero with the end-to-end continuous control adaptability of DreamerV3 and video-driven models. Efforts to unify transformer-based architectures with recurrent or diffusion approaches may also reshape capability landscapes.

Finally, more comprehensive, standardized benchmarks spanning physical realism, action diversity, and task complexity are needed to ensure consistent progress in constructing generalist, safe, and scalable world models. Transparency and interpretability of learned models and planning mechanisms remain important to enable trustworthy deployment in real-world applications.

These directions will continue to define world model research in the coming years.

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-05-09)
[2] Recurrent World Models Facilitate Policy Evolution. web. https://papers.nips.cc/paper/7512-recurrent-world-models-facilitate-policy-evolution.pdf (2018-12)
[3] Mastering Diverse Domains through World Models (DreamerV3). hf-search. https://huggingface.co/papers/2301.04104 (2023-01-10)
[4] Genie: Generative Interactive Environments. hf-search. https://huggingface.co/papers/2402.15391 (2024-02-23)
[5] MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. hf-search. https://huggingface.co/papers/1911.08265 (2019-11-19)
[6] ACWM-Phys: Investigating Generalized Physical Interaction in Action-Conditioned Video World Models. arxiv. https://arxiv.org/abs/2605.08567 (2026-05-09)
[7] DreamX-Phi 1.0: Action-Conditioned Video World Model for Robotic Manipulation. hf-search. https://huggingface.co/papers/2608.13489 (2026-08-13)
[8] τ₀-WM: A Unified Video-Action World Model for Robotic Manipulation. arxiv. https://arxiv.org/abs/2606.01027 (2026-06-03)
[9] Towards Interactive Video World Modeling (Survey). web. https://www.alphaxiv.org/abs/2606.01164 (2026-05-31)
