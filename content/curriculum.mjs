// The Fanin curriculum.
//
// Track accent colours are sampled along the viridis colormap — the palette
// every attention heatmap and loss surface in this field is already drawn in.
// `fill` is only ever used as a filled swatch (dot, ring, bar); `inkLight` and
// `inkDark` are the darkened/lightened variants safe to set as text colour.

export const site = {
  name: 'Fanin',
  tagline: 'The AI research curriculum, fully unlocked',
  description:
    'A complete, free path from linear algebra to distributed training. ' +
    '108 lessons across seven tracks. No accounts, no tiers, no paywall.',
  repo: 'https://github.com/gauransh18/fanin',
  license: 'CC BY-SA 4.0',
};

export const tracks = [
  {
    id: 'math',
    title: 'Mathematical Foundations',
    short: 'Math',
    fill: '#46327E',
    inkLight: '#46327E',
    inkDark: '#A192E4',
    blurb:
      'The linear algebra, calculus, probability and information theory that ' +
      'the rest of the curriculum assumes. Written for people who want to read ' +
      'papers, not pass an exam.',
    outcome: 'Read a method section and know what every symbol is doing.',
    lessons: [
      ['vectors-norms-geometry', 'Vectors, Norms, and Geometry', 14],
      ['matrices-as-linear-maps', 'Matrices as Linear Maps', 16],
      ['matrix-multiplication-cost', 'Matrix Multiplication and Its Cost', 15],
      ['rank-span-subspaces', 'Rank, Span, and the Four Subspaces', 16],
      ['eigenvalues-eigenvectors', 'Eigenvalues and Eigenvectors', 18],
      ['singular-value-decomposition', 'The Singular Value Decomposition', 20],
      ['derivatives-gradients-jacobians', 'Derivatives, Gradients, and Jacobians', 16],
      ['chain-rule-backprop', 'The Chain Rule, and Backprop by Hand', 20],
      ['hessians-and-curvature', 'Hessians, Curvature, and Second-Order Methods', 18],
      ['probability-spaces', 'Probability Spaces and Random Variables', 15],
      ['expectation-variance-concentration', 'Expectation, Variance, and Concentration', 17],
      ['distributions-in-ml', 'The Distributions That Actually Show Up', 16],
      ['bayes-and-independence', "Bayes' Rule and Conditional Independence", 15],
      ['maximum-likelihood', 'Maximum Likelihood and MAP', 17],
      ['entropy-cross-entropy-kl', 'Entropy, Cross-Entropy, and KL Divergence', 18],
      ['convexity-and-convergence', 'Convexity and Why Gradient Descent Works', 19],
    ],
  },
  {
    id: 'pytorch',
    title: 'Python and PyTorch',
    short: 'PyTorch',
    fill: '#3B528B',
    inkLight: '#3B528B',
    inkDark: '#93AAE8',
    blurb:
      'PyTorch as a systems tool, not a collection of incantations. Shapes, ' +
      'strides, autograd internals, and the training loop written the way it ' +
      'should be written.',
    outcome: 'Write, profile and fix a training loop without copying a template.',
    lessons: [
      ['tensors-dtypes-devices', 'Tensors: Dtypes, Devices, and Memory', 14],
      ['shapes-views-strides', 'Shapes, Views, and Strides', 18],
      ['broadcasting', 'Broadcasting, Precisely', 15],
      ['indexing-gather-scatter', 'Indexing, Gather, and Scatter', 16],
      ['einsum', 'einsum and Thinking in Indices', 17],
      ['autograd-graph', 'Autograd: The Graph You Never See', 19],
      ['custom-autograd-function', 'Writing a Custom autograd.Function', 17],
      ['nn-module', 'nn.Module and Parameter Management', 16],
      ['optimizers-schedulers', 'Optimizers and Learning-Rate Schedules', 18],
      ['datasets-dataloaders', 'Datasets, DataLoaders, and Collation', 16],
      ['training-loop', 'The Training Loop, Written Properly', 20],
      ['mixed-precision', 'Mixed Precision and Loss Scaling', 18],
      ['profiling', 'Profiling: Finding the Actual Bottleneck', 19],
      ['torch-compile', 'torch.compile and Graph Capture', 18],
      ['reproducibility', 'Seeds, Determinism, and Honest Benchmarks', 15],
    ],
  },
  {
    id: 'deep-learning',
    title: 'Deep Learning Core',
    short: 'Deep Learning',
    fill: '#2C728E',
    inkLight: '#256279',
    inkDark: '#74C8E0',
    blurb:
      'What makes deep networks trainable at all: initialization, normalization, ' +
      'residual paths, and the failure modes you will actually hit.',
    outcome: 'Diagnose why a network is not learning, from the loss curve alone.',
    lessons: [
      ['perceptron-linear-models', 'The Perceptron and Linear Models', 14],
      ['activation-functions', 'Activation Functions and Why They Differ', 16],
      ['mlps-universal-approximation', 'MLPs and Universal Approximation', 17],
      ['loss-functions', 'Loss Functions, and Choosing One', 18],
      ['initialization', 'Initialization: Xavier, He, and Why It Matters', 19],
      ['normalization', 'Normalization: Batch, Layer, and RMS', 20],
      ['regularization', 'Regularization: Decay, Dropout, Early Stopping', 18],
      ['residual-connections', 'Residual Connections and Deep Trainability', 18],
      ['convolutions', 'Convolutions and Inductive Bias', 19],
      ['pooling-receptive-fields', 'Striding, Pooling, and Receptive Fields', 16],
      ['recurrent-networks', 'Recurrent Networks and Vanishing Gradients', 18],
      ['lstm-gru', 'LSTMs, GRUs, and Gated Memory', 17],
      ['embeddings', 'Embeddings and Representation Learning', 17],
      ['autoencoders', 'Autoencoders and Latent Spaces', 18],
      ['generalization-double-descent', 'Generalization, Overfitting, Double Descent', 20],
      ['debugging-models', "Debugging a Model That Won't Learn", 22],
    ],
  },
  {
    id: 'transformers',
    title: 'Transformers and Attention',
    short: 'Transformers',
    fill: '#21918C',
    inkLight: '#19706C',
    inkDark: '#5AD6CE',
    blurb:
      'Attention derived from scratch, then built up into a working GPT, then ' +
      'taken apart again into the variants that make it fast.',
    outcome: 'Implement a transformer from an empty file, and explain every line.',
    lessons: [
      ['seq2seq-to-attention', 'From Seq2Seq to Attention', 17],
      ['scaled-dot-product-attention', 'Scaled Dot-Product Attention', 20],
      ['multi-head-attention', 'Multi-Head Attention', 18],
      ['positional-encodings', 'Positional Encodings: Sinusoidal, Learned, RoPE', 21],
      ['transformer-block', 'The Transformer Block', 18],
      ['encoder-decoder-architectures', 'Encoder, Decoder, and Encoder-Decoder', 17],
      ['causal-masking', 'Causal Masking and Teacher Forcing', 16],
      ['tokenization', 'Tokenization: BPE, WordPiece, SentencePiece', 20],
      ['gpt-from-scratch', 'Building a GPT from Scratch', 25],
      ['kv-caching', 'KV Caching', 18],
      ['attention-variants', 'Attention Variants: MQA, GQA, MLA', 19],
      ['flash-attention', 'FlashAttention and IO-Awareness', 20],
      ['long-context', 'Long Context: The Real Constraints', 19],
      ['mixture-of-experts', 'Mixture of Experts', 20],
      ['vision-transformers', 'Vision Transformers', 17],
      ['state-space-models', 'State Space Models and Mamba', 21],
    ],
  },
  {
    id: 'llms',
    title: 'Large Language Models',
    short: 'LLMs',
    fill: '#28AE80',
    inkLight: '#1C7A59',
    inkDark: '#57DAA8',
    blurb:
      'The full lifecycle: data, pretraining, scaling laws, alignment, reasoning, ' +
      'and the inference tricks that make any of it affordable.',
    outcome: 'Reason about a model card, a scaling decision, or an eval result.',
    lessons: [
      ['pretraining-objective', 'The Pretraining Objective', 18],
      ['pretraining-data', 'Data: Collection, Filtering, Deduplication', 21],
      ['scaling-laws', 'Scaling Laws and Compute-Optimal Training', 22],
      ['pretraining-infrastructure', 'Pretraining Infrastructure at Scale', 20],
      ['supervised-finetuning', 'Supervised Fine-Tuning', 18],
      ['peft-lora', 'Parameter-Efficient Fine-Tuning: LoRA and Friends', 20],
      ['reward-modeling', 'Reward Modeling from Human Preferences', 19],
      ['rlhf-ppo', 'RLHF with PPO', 22],
      ['dpo-direct-alignment', 'DPO and Direct Alignment', 20],
      ['reasoning-test-time-compute', 'Reasoning Models and Test-Time Compute', 21],
      ['decoding-strategies', 'Decoding: Greedy, Beam, Top-p, Temperature', 18],
      ['quantization', 'Quantization: INT8, INT4, and Below', 21],
      ['distillation', 'Distillation and Model Compression', 18],
      ['rag', 'Retrieval-Augmented Generation', 20],
      ['tool-use-agents', 'Tool Use and Agent Loops', 20],
      ['evaluating-llms', 'Evaluating LLMs Without Fooling Yourself', 22],
    ],
  },
  {
    id: 'rl',
    title: 'Reinforcement Learning',
    short: 'RL',
    fill: '#5EC962',
    inkLight: '#3D7F31',
    inkDark: '#8FE280',
    blurb:
      'From Bellman equations to PPO, built in the order the ideas actually ' +
      'depend on each other — and ending where RL meets language models.',
    outcome: 'Read an RL paper and place it on the policy/value, on/off-policy map.',
    lessons: [
      ['markov-decision-processes', 'Markov Decision Processes', 17],
      ['value-functions-bellman', 'Value Functions and the Bellman Equations', 19],
      ['dynamic-programming', 'Policy Iteration and Value Iteration', 18],
      ['monte-carlo-td', 'Monte Carlo and Temporal-Difference Learning', 19],
      ['q-learning-sarsa', 'Q-Learning and SARSA', 18],
      ['deep-q-networks', 'Deep Q-Networks', 20],
      ['policy-gradients', 'Policy Gradients and REINFORCE', 21],
      ['actor-critic', 'Actor-Critic and Advantage Estimation', 20],
      ['trpo-ppo', 'Trust Regions: TRPO and PPO', 22],
      ['ddpg-td3-sac', 'Off-Policy Actor-Critic: DDPG, TD3, SAC', 21],
      ['exploration', 'Exploration: From ε-Greedy to Curiosity', 18],
      ['model-based-rl', 'Model-Based RL and World Models', 20],
      ['offline-rl', 'Offline RL', 19],
      ['rl-for-language-models', 'RL for Language Models', 20],
    ],
  },
  {
    id: 'systems',
    title: 'Systems and MLOps',
    short: 'Systems',
    fill: '#AADC32',
    inkLight: '#5E7A15',
    inkDark: '#C9E863',
    blurb:
      'Where the arithmetic meets the hardware: GPU memory hierarchies, custom ' +
      'kernels, sharding strategies, serving, cost, and reproducing papers.',
    outcome: 'Size a training run, shard it correctly, and serve it efficiently.',
    lessons: [
      ['gpu-architecture', 'GPU Architecture for ML Engineers', 20],
      ['memory-hierarchy-roofline', 'Memory Hierarchy and Arithmetic Intensity', 21],
      ['cuda-kernels', 'Writing a Custom CUDA Kernel', 22],
      ['triton', 'Triton and Fused Kernels', 21],
      ['data-parallelism', 'Data Parallelism and All-Reduce', 20],
      ['model-parallelism', 'Tensor, Pipeline, and Sequence Parallelism', 22],
      ['zero-fsdp', 'ZeRO and Fully Sharded Data Parallel', 21],
      ['gradient-checkpointing', 'Gradient Checkpointing and Memory Budgets', 18],
      ['inference-serving', 'Inference Serving: Batching and Throughput', 21],
      ['speculative-decoding', 'Speculative Decoding', 19],
      ['experiment-tracking', 'Experiment Tracking and Reproducible Research', 17],
      ['testing-ml-code', 'Testing Machine Learning Code', 19],
      ['deploying-monitoring', 'Deploying and Monitoring Models', 19],
      ['cost-modeling', 'Cost Modeling for Training Runs', 18],
      ['reading-reproducing-papers', 'Reading and Reproducing Papers', 20],
    ],
  },
];

// Flat, ordered lesson list with back-references to the owning track.
export const allLessons = tracks.flatMap((track, ti) =>
  track.lessons.map(([slug, title, minutes], li) => ({
    slug,
    title,
    minutes,
    trackId: track.id,
    trackTitle: track.title,
    trackShort: track.short,
    trackIndex: ti,
    indexInTrack: li,
    number: `${ti + 1}.${String(li + 1).padStart(2, '0')}`,
    href: `/learn/${track.id}/${slug}/`,
  }))
);

export const totals = {
  tracks: tracks.length,
  lessons: allLessons.length,
  minutes: allLessons.reduce((sum, l) => sum + l.minutes, 0),
};
