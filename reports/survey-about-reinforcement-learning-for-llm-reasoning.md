# Survey on Reinforcement Learning for Large Language Model Reasoning

## TL;DR
- Foundational methods such as STaR (Self-Taught Reasoner) innovate iterative rationale bootstrapping for LLM reasoning, with RL-STaR providing a theoretical RL framework supporting policy improvement through chain-of-thought generation and convergence guarantees [1][2].
- Recent core RL algorithms for reasoning like Group Relative Policy Optimization (GRPO) improve training efficiency by removing value models and relying on group-normalized rewards, effectively used in state-of-the-art frameworks like DeepSeek-R1 [3][4].
- Reinforcement Learning with Verifiable Rewards (RLVR) leverages deterministic reward signals such as test correctness to boost reasoning capabilities and encourage emergent reasoning behaviors without supervised rationales [5][6][7].
- Process Reward Models (PRMs) provide fine-grained, stepwise evaluation of intermediate reasoning steps, improving credit assignment and training stability for reasoning LLMs [8][9][10].
- Advanced policy optimization strategies address exploration-exploitation trade-offs and reasoning efficiency through rollout dropout (GRPODropout), adaptive reasoning timing (RACE), and agentic RL tool integration, collectively enhancing robustness and scalability of LLM reasoning [11][12][13].
- Challenges remain in reward signal design, training stability, empirical generalization, and integrating multi-modal reasoning and tool use under unified RL frameworks.

## Background
Reinforcement learning (RL) has emerged as a powerful paradigm for enhancing reasoning capabilities in large language models (LLMs). Early approaches mainly relied on supervised fine-tuning with chain-of-thought (CoT) data, but these lacked the ability to self-improve or leverage outcome-based feedback effectively [1]. STaR (Self-Taught Reasoner) introduced a foundational method for iterative improvement by generating and filtering rationales that yield correct answers, enabling LLMs to bootstrap their reasoning ability through self-generated data and fine-tuning [1]. RL-STaR developed a formal reinforcement learning framework grounded in policy improvement guarantees for such iterative rationale training despite noisy data [2].

Since then, RL algorithms specifically adapted to LLM reasoning have proliferated, with a focus on efficient signal utilization, robust credit assignment, and exploration-exploitation trade-offs. Group Relative Policy Optimization (GRPO) eliminated the need for a separate value-function critic by leveraging normalized group rewards per prompt, reducing compute cost and memory footprints while maintaining strong policy updates [3]. Concurrently, RLVR (Reinforcement Learning with Verifiable Rewards) offered a principled way to apply verifiable correctness-based reward signals—such as unit tests or deterministic checks—making reinforcement learning more reliable and interpretable in reasoning domains like math and coding [5][6][14][7]. 

Process Reward Models (PRMs) extended this paradigm by shifting from outcome-only feedback to dense, stepwise evaluation of intermediate reasoning steps, improving alignment and sample efficiency. Frameworks like PRL (Process Reward Learning) implement entropy-regularized RL objectives decomposed into fine-grained process rewards, supported by theoretical backing and empirical gains on benchmarks [8][9][10].

Advances in policy optimization include rollout management techniques like GRPODropout to maintain exploration diversity [11], adaptive reasoning policies deciding when to reason (RACE) [12], and agentic reinforcement learning integrating tool use dynamically (ARTIST) [13], all contributing to scalable and robust LLM reasoning enhancement.

## Iterative Bootstrap and Self-Taught Reasoning: STaR and RL-STaR
The Self-Taught Reasoner (STaR) represents a foundational method that enables LLMs to bootstrap their reasoning capabilities by generating rationales internally and filtering them for correctness before fine-tuning. This cyclical approach bypasses the need for large annotated chain-of-thought datasets and enables continual improvement.

STaR’s training loop involves:
1. Prompting the LLM to produce chain-of-thought rationales and answers on a set of problems with few-shot demonstrations.
2. Selecting generated outputs where the final answers are correct.
3. Regenerating rationales conditioned on the correct answers for failed attempts.
4. Fine-tuning the model on aggregated rationales that yield correct reasoning.

This iterative rationale bootstrapping is significant because it leverages the model’s own partial capabilities to self-improve, demonstrated by performance gains comparable to fine-tuning models 30 times larger [1]. RL-STaR extends STaR by framing this process within a reinforcement learning policy improvement paradigm, mathematically demonstrating that even with noisy reasoning trajectories an RL policy converges towards optimal chains of thought under reasonable assumptions [2]. The formalism uses concepts of reward signaling, transition probabilities of reasoning steps, and policy evaluation to validate empirical STaR results.

## Efficient Reinforcement Learning Algorithms: GRPO and DeepSeek-R1
Group Relative Policy Optimization (GRPO) is a recent core algorithm designed to improve the efficiency of RL training for reasoning LLMs by removing the necessity for a separate critic (value function). Instead, it calculates normalized advantages based on the relative quality of answer groups generated per query prompt [3]. This strategy reduces training complexity and memory usage, particularly important when scaling RL to large language models.

DeepSeek-R1 operationalizes these insights to train LLMs on reasoning tasks such as mathematics, coding, and logic. It implements a multi-stage training process including cold-start data pre-fine-tuning, rejection sampling, RL training with GRPO, and supervised fine-tuning, guided by rule-based rewards for reasoning correctness plus model-based rewards for general language consistency and harmlessness [4]. This staged approach enables emergent reasoning capabilities like self-reflection, verification, and dynamic strategy adjustment, notably without the supervision of reasoning step labels. Avoiding neural reward models for reasoning tasks reduces risks of reward hacking.

GRPO’s group normalization mechanism also addresses biases in sequence length and difficulty, with ongoing refinements like Dr. GRPO removing normalizations that induced reward biases [3][4]. As a result, DeepSeek-R1 attains performance on reasoning benchmarks comparable to leading models including those from major research labs, while sustaining training efficiency.

## Reinforcement Learning with Verifiable Rewards (RLVR) and Process Reward Models
The RLVR paradigm enhances reinforcement learning for reasoning by employing verifiable and deterministic reward signals derived from correctness checks, such as test suites for code, calculators for math, or formal proofs in logic [5][6][14]. Unlike reward models learned from human preferences, RLVR uses objective, verifiable outcomes to guide policy updates, improving interpretability and stability.

RLVR methods incentivize LLMs to generate not only correct final answers but also reasoned solution paths featuring properties like self-verification and dynamic adaptation [7]. Empirical work has demonstrated that even one-shot RL with verifiable rewards can induce strong generalization and continual improvement after accuracy saturation, known as post-saturation generalization [5].

Process Reward Models (PRMs) extend verifiable rewards into intermediate reasoning steps, assigning dense credit signals that improve learning efficiency and stability in reinforcement learning [8][9]. PRMs are trained on annotated or bootstrapped intermediate reasoning steps to provide stepwise feedback during training, bridging the gap between sparse outcome-only rewards and fully supervised stepwise labels.

Recent frameworks such as Process Reward Learning (PRL) implement entropy-regularized reinforcement learning objectives decomposed into stepwise rewards, improving the policy learning process and expanding the reasoning boundary for LLMs on math and logic tasks [10]. These improvements facilitate fine-grained credit assignment and reduce sample inefficiency, which are critical in complex multi-step reasoning scenarios.

## Policy Optimization Advances: Exploration, Adaptive Reasoning, and Agentic RL
Recent advances in policy optimization for LLM reasoning address common challenges such as exploration collapse, reasoning efficiency, and complex multi-modal decision making.

GRPODropout modifies the standard GRPO by selectively removing a small subset of high-probability advantageous rollouts before policy updates. This counteracts policy entropy collapse, prevents overfitting to limited trajectories, and preserves diversity essential for exploration in reasoning tasks, yielding higher accuracy and entropy with less computational overhead [11].

RACE introduces adaptive reasoning policies that learn when to perform reasoning during sequential multi-turn tasks, reducing redundant reasoning and lowering inference costs without sacrificing performance [12]. By assessing the utility of reasoning steps across interaction turns, RACE trains policies that economize reasoning effort dynamically.

ARTIST integrates reinforcement learning with agentic reasoning and dynamic tool invocation, empowering LLMs to decide autonomously when and how to trigger external tools during multi-turn reasoning. This enhances solution quality and enables deeper interaction with environments, expanding traditional reward models to account for tool use and multi-action policies [13].

Other methods, such as the T1 RL approach, combine chain-of-thought pretraining with oversampling strategies and entropy bonuses, demonstrating superior test-time scaling and reward optimization for complex reasoning [15]. Additionally, residual advantage refinements redistribute credit at granular token steps, improving policy credit assignment within RL frameworks [16]. Reverse-engineered reasoning (REER) techniques complement RL by providing structured reasoning trajectory datasets that enrich training and reward modeling, promoting deeper reasoning capabilities [17].

## Trends and open problems
The ongoing evolution of reinforcement learning methods for LLM reasoning exhibits a multifaceted approach toward improving reasoning capability, efficiency, and generalization. Foundational methods like STaR have introduced self-bootstrapping rationale generation, while algorithmic advances like GRPO enable more efficient and scalable RL training. The emergence of RLVR and PRM techniques underscore the importance of verifiable and process-level reward signals over traditional preference or outcome-only rewards.

However, challenges remain. Designing robust reward functions that balance correctness, readability, and interpretability continues to be difficult. Further research is needed to address reward hacking and training stability, especially in large-scale and multi-domain contexts. Empirical benchmarks evaluating the comparative effectiveness of different RL algorithms and reward formulations for diverse reasoning tasks remain sparse.

Integration of RL methods with adaptive reasoning frequency control, tool use, and multi-modal knowledge retrieval calls for unifying frameworks that can jointly optimize policy and reward models over heterogeneous action spaces. Finally, ensuring RL methods scale and generalize across both small and large LLMs, diverse reasoning domains, and practical applications is an imperative avenue for future research.

This survey highlights the promising advances in reinforcement learning tailored for LLM reasoning and indicates key trajectories to further mature and deploy these techniques effectively.

## References
[1] STaR: Self-Taught Reasoner. web. https://proceedings.neurips.cc/paper_files/paper/2022/file/639a9a172c044fbb64175b5fad42e9a5-Paper-Conference.pdf (2022)
[2] RL-STaR: Theoretical Analysis of Reinforcement Learning Frameworks for Self-Taught Reasoner. arxiv. https://arxiv.org/abs/2410.23912 (2025-04-10)
[3] The State of Reinforcement Learning for LLM Reasoning. web. https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training (2025-04-19)
[4] Reinforcement Learning for Reasoning in Large Language Models. arxiv. https://arxiv.org/abs/2504.20571 (2025-04)
[5] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. hf-search. https://huggingface.co/papers/2501.12948 (2025-01-22)
[6] RLVR-World and RLVR Related Techniques. hf-search. https://huggingface.co/papers/2505.13934 (2025-05-20)
[7] The Lessons of Developing Process Reward Models in Mathematical Reasoning. hf-search. https://huggingface.co/papers/2501.07301 (2025-01-13)
[8] What is RLVR? Reinforcement Learning from Verifiable Rewards. web. https://rlvrbook.com/ (n.d.)
[9] Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs - Microsoft Research. web. https://www.microsoft.com/en-us/research/publication/reinforcement-learning-with-verifiable-rewards-implicitly-incentivizes-correct-reasoning-in-base-llms/ (2026-02-17)
[10] A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for Large Language Models. web. https://exa.ai/library/publication/gbf7ftjb3j8 (2025-10-09)
[11] PRL: Process Reward Learning Improves LLMs' Reasoning. arxiv. https://arxiv.org/abs/2601.10201 (2026-01-26)
[12] GRPODropout: Less is More for Online Reinforcement Learning Rollouts. arxiv. https://arxiv.org/abs/2610.11854 (2026-10-08)
[13] When Should Agents Think? Adaptive Reasoning via Cross-Turn Estimation. arxiv. https://arxiv.org/abs/2610.12061 (2026-10-08)
[14] Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling. hf-search. https://huggingface.co/papers/2501.11651 (2025-01-20)
[15] Agentic Reasoning and Tool Integration for LLMs via Reinforcement Learning. hf-search. https://huggingface.co/papers/2505.01441 (2025-04-28)
[16] Residual Advantage: Student-Relative Teacher Guidance for RL with Verifiable Rewards. arxiv. https://arxiv.org/abs/2610.11519 (2026-10-08)
[17] Reverse-Engineered Reasoning for Open-Ended Generation. hf-search. https://huggingface.co/papers/2509.06160 (2025-09-07)
