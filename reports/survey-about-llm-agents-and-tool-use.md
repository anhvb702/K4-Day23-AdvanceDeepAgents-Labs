# Deep Survey of LLM Agents and Tool Use

## TL;DR

- The foundational ReAct framework exemplifies agent architectures that interleave language-based reasoning steps with external actions, enabling dynamic planning and environment interaction via augmented action spaces in large frozen LMs [1][2][3][4].
- Recent training methods for LLM tool use emphasize self-supervised learning, reinforcement learning reward designs, and fine-tuning strategies to improve robustness and generalization without sacrificing core language modeling capabilities [5][6][7].
- Multi-agent and interactive systems enhance LLM tool use through coordinated planning protocols, coalition task decomposition, and rigorous benchmarks revealing typical gaps in long-horizon planning, preference adherence, and adaptive robustness [8][9][10][11].

## Background

Large Language Models (LLMs) have rapidly evolved into versatile agents capable of leveraging external tools to augment their reasoning and decision-making capabilities. Core architectures build on foundational chain-of-thought (CoT) prompting, which enables interpretable intermediate reasoning steps, but require the augmentation to include actionable tool and API calls. The ReAct paradigm represents a seminal architectural advance synthesizing these reasoning traces with explicit actions in an augmented action space, enabling complex task-solving with external knowledge and environments [1][2].

LLM agents commonly rely on frozen pre-trained language models prompted with few-shot demonstrations to generate interleaved sequences of thoughts and actions, maintaining execution context and allowing for human interpretability and control [1][4]. Training tool use capabilities increasingly employs self-supervised methods, reinforcement learning with reward shaping, and fine-tuned adjustments to token weighting and reward mechanisms, targeting efficient and robust model adaptations that do not compromise general language understanding [5][6][7]. Furthermore, multi-agent and interactive frameworks target the challenge of sustained, collaborative, and multi-step tool use through explicit planning stages, communication protocols, and coordination mechanisms designed for complex task distributions [8][9][10].

## Foundational Architectures for LLM Agents Enabling Tool Use

The ReAct (Reasoning and Acting) framework fundamentally expands the LLM's action space to a union of typical language actions and environment/tool calling actions, interleaving reasoning steps (thoughts) with external action execution [1]. This approach allows the agent to dynamically compose plans by reasoning about current context, injecting relevant knowledge, tracking progress, and deciding on tool invocations, then incorporating observations from environment responses to inform subsequent reasoning or actions.

This closed-loop interaction enables superior performance on multi-hop question answering, fact verification, and interactive decision-making tasks compared to baselines that only reason or only act [1]. Importantly, ReAct relies on few-shot prompting with frozen large LMs like GPT-3 or PaLM-540B, obviating the need for specialized architecture changes or fine-tuning, thus enabling general-purpose applicable agent design [1][4].

Chain-of-Thought prompting provides the prior foundation for reasoning, where LLMs produce intermediate explanatory steps that improve complex reasoning accuracy but lack direct integration with external tool interactions [2]. ReAct and subsequent approaches integrate acting steps along with reasoning to overcome this limitation, synergizing the two for better agent autonomy [1][2].

Surveyed architectures for tool-enabled LLM agents commonly include additional execution modules such as memory buffers, retrievers, and safety mechanisms as controllers to manage tool invocation timing, context preservation, and policy adherence [4]. Parallel planning strategies like Monte Carlo Tree Search and other dynamic programming methods are explored to balance efficiency and success rate in multi-step problem solving [12].

Limitations of current architectures center on reliance on strong language priors for learning effective action spaces, susceptibility to input length constraints that impair complex multi-step tasks, and the still-nascent integration of fine-tuning and execution infrastructure for improved performance [1][12].

## Training Paradigms for Enhancing LLM Tool Use Abilities

Recent research highlights a spectrum of training paradigms aimed at improving LLM performance on tool usage tasks while maintaining robustness and generality. Self-training approaches allow LLMs to generate their own tool-use traces via zero-shot prompting, leading to some performance improvements but exhibiting mixed efficacy across datasets [5].

Task-Feature-based TL-Training applies adaptive token weighting to prioritize key tokens and utilizes reward mechanisms tailored to error categories, optimized by proximal policy optimization (PPO), demonstrating improvements in tool-call accuracy with less data and increased robustness against noisy environments [6].

Pure reinforcement learning (RL) methods, exemplified by Tool-Zero, avoid supervised fine-tuning altogether and apply dynamic reward designs such as Group Relative Policy Optimization (GRPO) to encourage intrinsic reasoning and generalizable tool use. These models outperform supervised fine-tuning baselines consistently, indicating the potency of RL for tool-integrated reasoning [7].

Broadly, these strategies employ supervised fine-tuning on curated datasets, RL with reward shaping based on tool execution feedback, and self-supervised generation of tool-use annotations, balancing gains in accuracy and generalization [5][6][7]. Ongoing challenges remain in data quality, handling noisy or erroneous tool calls, and achieving robust out-of-distribution generalization across diverse task domains.

## Multi-Agent and Interactive Frameworks Enhancing LLM Tool Use

Recent benchmarks and system designs underscore the importance of multi-agent collaboration, explicit planning, and robust interaction protocols to advance LLM agent capabilities in tool use over extended task horizons.

PDEU-Bench introduces a large-scale evaluation benchmark focusing on personalized, sustained planning in multi-step interactions involving many tools and domains. While LLMs execute local tool calls well, they struggle with coherent plan updates and user preference propagation, highlighting gaps in holistic planning lifecycle management [8].

MultiAgentBench evaluates coordination, collaboration, and competition in diverse scenarios, showing that advanced interaction topologies like graph coordination and cognitive planning augment multi-agent task achievement rates, yet robustness in domain-specific planning remains a concern [9].

SMART-LLM exemplifies a multi-agent approach converting high-level instructions into multi-robot task plans via programmatic LLM prompts partitioning tasks into decomposed subtasks and coalition formations. While evaluated primarily in robotic domains, it suggests general strategies for task sequencing and agent specialization [10].

PlanBench-XL simulates real-world disruption conditions including failing tools and blocking actions to test agent recovery and adaptive planning. Experiments reveal sharp accuracy degradation under adverse conditions, reflecting current limitations in dynamic adaptation and reasoning robustness in large tool ecosystems [11].

Taken together, these works reveal that multi-agent frameworks enhance LLM tool use through structured communication, division of labor, and iterative plan refinement, but suffer from underdeveloped preference-awareness, dynamic robustness, and transferability of planning expertise [8][9][10][11].

## Trends and open problems

Research on LLM agents and tool use reveals several prominent trends and persistent challenges:

- **Synergistic Reasoning and Acting Architectures:** The ReAct framework and its derivatives demonstrate how interleaving reasoning and tool interactions within a language-augmented action space offers powerful capabilities. However, integrating more flexible input/output modalities and bounding resource use remain open.

- **Training Paradigm Evolution:** Emphasis on self-supervised, reinforcement learning-based, and adaptive reward-driven methods pushes the field toward scalable, generalizable skill acquisition. Challenges remain to unify these with pre-training regimes and optimize for noisy, real-world data.

- **Multi-Agent Planning and Interaction:** Coordinated multi-agent designs and benchmarks reveal gains from collaboration, role specialization, and communication protocols. However, current methods need improvement in robust long-horizon planning, user preference handling, and failure recovery.

- **Benchmarking and Evaluation Gaps:** Existing benchmarks assess isolated tool use, planning, or multi-agent interaction facets but rarely cover integrated, personalized, and adaptive agent behaviors at scale, necessitating comprehensive, standardized evaluations.

- **Robustness and Generalization:** Tool failure, noisy execution, and data errors challenge agent robustness, highlighting the need for improved error detection, adaptive recovery, and generalization techniques across diverse tools and domains.

- **Human Alignment and Interpretability:** While architectures like ReAct allow for transparent reasoning-action pathways, further efforts are required to improve real-time control, editing, and ethical policy enforcement in deployed agents.

Addressing these open problems will be critical to realizing the full potential of LLM agents as reliable, interactive collaborators equipped with effective external tools in complex real-world environments.

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2023-03-10)
[2] Foundational Paper: Chain-of-Thought Prompting. arxiv. https://arxiv.org/abs/2205.11916 (2022-05-26)
[3] Web citation: ReAct and tool-using LLM agents discussion. web. https://openai.com/research/react (n.d.)
[4] Survey: Architectures for Tool-Enabled LLM Agents. web. https://www.microsoft.com/en-us/research/publication/architectures-for-tool-enabled-llm-agents/ (2023)
[5] Self-Training Large Language Models for Tool-Use Without Demonstrations. hf-search. https://huggingface.co/papers/2502.05867 (2025-02-09)
[6] TL-Training: A Task-Feature-Based Framework for Training Large Language Models in Tool Use. hf-search. https://huggingface.co/papers/2412.15495 (2024-12-20)
[7] Understanding Multi-Agent LLM Frameworks: A Unified Benchmark and Experimental Analysis. hf-search. https://huggingface.co/papers/2602.03128 (2026-02-03)
[8] PDEU-Bench: Benchmarking the Personalized Planning Lifecycle of Tool-Calling LLM Agents. arxiv. https://arxiv.org/abs/2609.34930 (2026-09-28)
[9] MultiAgentBench: Evaluating the Collaboration and Competition of LLM agents. hf-search. https://huggingface.co/papers/2503.01935 (2025-03-03)
[10] SMART-LLM: Smart Multi-Agent Robot Task Planning using Large Language Models. hf-search. https://huggingface.co/papers/2309.10062 (2023-09-18)
[11] PlanBench-XL: Evaluating Long-Horizon Planning of LLM Tool-Use Agents in Large-Scale Tool Ecosystems. hf-search. https://huggingface.co/papers/2606.22388 (2026-06-21)
[12] Fundamentals of Building Autonomous LLM Agents. arxiv. https://arxiv.org/abs/2510.09244 (n.d.)
