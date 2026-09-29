const QUESTIONS = [
  // ── Transformer Architecture ──────────────────────────────────────────────
  {
    id: "tf-1",
    theme: "Transformer Architecture",
    question: "Explain the Transformer architecture end-to-end.",
    diagram: "transformer",
    answer:
`[Glassdoor confirmed: "Explain transformers in the context of LLMs"]

• Input: token IDs → embedding lookup + positional encoding (summed, not concatenated)
• Encoder block (per layer): Multi-Head Self-Attention → Add & Norm → FFN → Add & Norm
  - All positions attend to all positions (bidirectional)
• Decoder block (per layer): Masked MHA (causal) → Cross-Attention (Q=decoder, K/V=encoder) → FFN; each sublayer wrapped in residual + LN
• Scaled dot-product attention: Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V
• Multi-head: h parallel heads, each with W_Q, W_K, W_V ∈ ℝ^{d_model × dₖ}; outputs concatenated then projected: MultiHead = Concat(head₁…headₕ) · W_O
• FFN: two-layer MLP with nonlinearity — ReLU in original, GeLU or GeGLU in modern LLMs
• Output (decoder): linear layer projects to vocab size → softmax for next-token probability`
  },
  {
    id: "tf-2",
    theme: "Transformer Architecture",
    question: "What is scaled dot-product attention? Why do we scale by √dₖ?",
    answer:
`• Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V
• Why scale: Q and K each have dimension dₖ; their dot product has variance ≈ dₖ assuming zero-mean unit-variance components
• Without scaling: large dₖ → large dot products → softmax saturates toward one-hot → near-zero gradients on most logits
• Dividing by √dₖ restores variance to ~1, keeping softmax in the well-behaved gradient regime regardless of head dimension`
  },
  {
    id: "tf-3",
    theme: "Transformer Architecture",
    question: "What does multi-head attention gain over single-head attention?",
    answer:
`• Each head projects Q, K, V to lower dimension dₖ = d_model/h, running h attention functions in parallel
• Allows the model to jointly attend to information from different representation subspaces at different positions — each head can specialize (e.g. one head tracks syntax, another coreference)
• Single-head attention collapses all information into one subspace; multi-head avoids this bottleneck
• MultiHead(Q,K,V) = Concat(head₁, …, headₕ) W_O where headᵢ = Attention(QWᵢ_Q, KWᵢ_K, VWᵢ_V)`
  },
  {
    id: "tf-4",
    theme: "Transformer Architecture",
    question: "How does positional encoding work in the original Transformer? What are modern alternatives?",
    answer:
`• Sinusoidal (original): PE(pos, 2i) = sin(pos / 10000^{2i/d}), PE(pos, 2i+1) = cos(…); no learned params; added to input embeddings; allows some extrapolation in theory
• Learned absolute PE (BERT, GPT-2): trainable embedding table indexed by position; cannot extrapolate beyond training length
• RoPE (Rotary PE — Llama, Mistral, Gemma): encodes position by rotating Q and K in the complex plane; relative distances are preserved in dot products; extrapolates better than absolute PE; dominant in current LLMs
• ALiBi: adds a fixed linear bias (−|i−j| × slope) to attention logits; no positional modification to embeddings; strong length generalization`
  },
  {
    id: "tf-5",
    theme: "Transformer Architecture",
    question: "Compare encoder-only, decoder-only, and encoder-decoder Transformers with examples.",
    answer:
`• Encoder-only (BERT, RoBERTa): bidirectional self-attention (all positions attend to all); pretrained with masked language modeling; best for classification, NER, embedding, and retrieval
• Decoder-only (GPT, Llama, Gemma): causal/autoregressive mask — each token attends only to past tokens; trained on next-token prediction; default architecture for generative LLMs
• Encoder-decoder (T5, BART, original Transformer): encoder produces contextualized representations of input; decoder attends via cross-attention over encoder output; natural for seq2seq tasks (translation, summarization)`
  },
  {
    id: "tf-6",
    theme: "Transformer Architecture",
    question: "What are Gated Linear Units (GLU) and GeGLU? Why are they preferred over ReLU/GeLU in modern LLMs?",
    answer:
`• Standard FFN: FFN(x) = activation(xW₁) W₂ — single linear transform then nonlinearity
• GLU: introduces a second "gate" branch: FFN_GLU(x) = (xW₁) ⊗ activation(xW₂); the gate dynamically controls how much of each dimension passes through based on input context
• GeGLU specifically: gate uses GeLU: FFN_GeGLU(x) = (xW₁) ⊗ GeLU(xW₂); output projected via W₃ to residual stream
• Why preferred: gating lets the model suppress or amplify features dynamically — more expressive than a static nonlinearity; empirically shows faster convergence and better performance per parameter
• Tradeoff: 3 weight matrices (W₁, W₂, W₃) instead of 2 — slight parameter overhead, typically offset by reducing hidden dim proportionally
• Used in: Llama 2/3, Gemma, PaLM — the standard FFN variant in current high-performance LLMs`
  },

  // ── LLM Training Pipeline ─────────────────────────────────────────────────
  {
    id: "train-1",
    theme: "LLM Training Pipeline",
    question: "Walk through the full LLM training pipeline from pretraining to deployment.",
    answer:
`[Glassdoor confirmed: "What is the process of training transformers — supervised finetuning, RL involved?"]

1. Pretraining: next-token prediction on massive web corpus (trillions of tokens); model learns language, world knowledge, and basic reasoning
2. SFT (Supervised Fine-Tuning): teacher-forcing on (prompt, ideal response) demonstration pairs; teaches instruction-following format
3. Reward Modeling: train RM on human preference pairs (A vs B); uses Bradley-Terry pairwise loss: log σ(r(x,y_w) − r(x,y_l))
4. RLHF via PPO: maximize E[r(x,y)] − β·KL(π_θ || π_ref); KL term penalizes deviation from SFT model, preventing reward hacking
   OR DPO: skip RM entirely; direct supervised loss on (prompt, chosen, rejected) triples
5. Evaluation: benchmarks (MMLU, HumanEval, MT-Bench) + red-teaming + human preference evals`
  },
  {
    id: "train-2",
    theme: "LLM Training Pipeline",
    question: "What is SFT (Supervised Fine-Tuning)? What data and loss does it use?",
    answer:
`• Fine-tune a pretrained LLM on demonstration data: (prompt, ideal response) pairs from human contractors or distillation from stronger models
• Loss: cross-entropy on response tokens only — prompt tokens are masked out; only the assistant's answer is supervised
• Training regime: teacher forcing — the ground truth token is fed as input at each step regardless of model predictions
• Purpose: shifts behavior from "predict internet text" to "follow instructions as a helpful assistant"
• Data quality over quantity: a few thousand high-quality demonstrations outperform millions of low-quality ones`
  },
  {
    id: "train-3",
    theme: "LLM Training Pipeline",
    question: "Explain RLHF end-to-end. What is the reward model and how is PPO applied?",
    answer:
`1. Collect preference data: for each prompt, sample two model outputs; human labeler picks the preferred one
2. Train reward model r_φ(x,y): Bradley-Terry loss: L = −E[log σ(r(x,y_w) − r(x,y_l))]
   - r_φ is typically the SFT model with the final LM head replaced by a scalar regression head
3. Fine-tune LLM with PPO:
   • Sample prompt x; generate response y ~ π_θ
   • Score with r_φ; compute advantages using GAE
   • Maximize: E[r_φ(x,y)] − β · KL(π_θ(y|x) || π_ref(y|x))
   • KL coefficient β prevents reward hacking — the policy can't exploit loopholes in r_φ while staying near π_ref
4. Four models in memory: policy π_θ, frozen reference π_ref, reward model r_φ, value network`
  },
  {
    id: "train-4",
    theme: "LLM Training Pipeline",
    question: "What is DPO and how does it differ from RLHF with PPO?",
    answer:
`• Direct Preference Optimization (Rafailov et al. 2023): eliminates the explicit reward model and RL training loop entirely
• Derives a closed-form supervised loss directly from the RLHF objective under the optimal policy:
  L_DPO = −E[ log σ( β log(π_θ(y_w|x)/π_ref(y_w|x)) − β log(π_θ(y_l|x)/π_ref(y_l|x)) ) ]
• Train directly on (prompt, chosen, rejected) triples with the same memory footprint as SFT
• Advantages: simpler pipeline, more stable training, no reward hacking, no separate reward model
• Tradeoffs: offline only (no online exploration); slightly less flexible than PPO; can degrade on out-of-distribution prompts`
  },
  {
    id: "train-5",
    theme: "LLM Training Pipeline",
    question: "What is LoRA? How does it work and when do you prefer it over full fine-tuning?",
    answer:
`• Low-Rank Adaptation: freeze all base model weights; inject trainable ΔW = B·A into attention projections (A ∈ ℝ^{r×k}, B ∈ ℝ^{d×r}, r ≪ min(d,k))
• Init: A ~ N(0,σ), B = 0 so ΔW = 0 at training start; the model begins from its pretrained behavior
• Forward: h = W₀x + (α/r) · BAx; α is a fixed scaling hyperparameter
• Typical r = 8–64; reduces trainable parameters by 100–1000× vs full fine-tuning
• Prefer LoRA when: GPU memory is limited, multiple task adapters needed, or fast iteration required
• QLoRA: quantize base model to 4-bit NF4, keep LoRA adapters in bf16 → fine-tune 70B+ models on a single A100`
  },
  {
    id: "train-6",
    theme: "LLM Training Pipeline",
    question: "How does knowledge distillation work, and why is it effective for training smaller LLMs?",
    answer:
`• A large "teacher" model generates soft probability distributions over the full vocabulary; the "student" model is trained to minimize KL-divergence between its output distribution and the teacher's, rather than cross-entropy against hard one-hot labels
• Key insight: soft labels encode the teacher's uncertainty and token relationships (near-synonyms get similar probability mass) — richer training signal than hard targets
• Temperature scaling: teacher logits are divided by T > 1 before softmax to soften distributions, amplifying signal from non-top tokens
• Why effective: the student learns not just "what is the right answer" but "how confident to be and what alternatives exist" — compresses the teacher's generalization into fewer parameters
• Used in: Gemma 3 (distilled from Gemini), Llama 4 Maverick (distilled from Behemoth), DeepSeek-R1 (reasoning capabilities distilled back into V3)`
  },

  // ── Attention & Efficiency ────────────────────────────────────────────────
  {
    id: "attn-1",
    theme: "Attention & Efficiency",
    question: "What is the computational and memory complexity of self-attention? Why is it a bottleneck?",
    answer:
`• Time: O(n²·d) — computing QKᵀ requires n² dot products each of dimension d
• Memory: O(n²) — the full n×n attention matrix must be stored for the backward pass
• Bottleneck: quadratic scaling with sequence length n; 4K tokens → 16M scores; 128K tokens → 16B scores
• Motivates: FlashAttention (memory-efficient exact attention), sparse/local attention, linear attention approximations, and state-space models (Mamba) that achieve O(n) in sequence length`
  },
  {
    id: "attn-2",
    theme: "Attention & Efficiency",
    question: "What is FlashAttention and how does it achieve memory efficiency?",
    answer:
`• IO-aware exact attention algorithm (Dao et al. 2022) — mathematically identical to standard attention, just implemented differently
• Key insight: GPU HBM (main memory) is slow; on-chip SRAM is fast but tiny; standard attention writes the O(n²) matrix to HBM
• FlashAttention tiles Q, K, V into SRAM blocks and fuses softmax + matmul in a single kernel — the full n×n matrix is never materialized in HBM
• Online softmax trick: computes numerically stable softmax incrementally across blocks, accumulating the correct result without storing all logits simultaneously
• Memory: O(n) instead of O(n²); arithmetic is identical to standard attention
• Speedup: 2–4× on A100 from reduced memory bandwidth; enables long-context training that was previously infeasible`
  },
  {
    id: "attn-3",
    theme: "Attention & Efficiency",
    question: "What is the KV cache? Why is it critical for autoregressive inference?",
    answer:
`• During autoregressive generation, K and V tensors for all past tokens are fixed and identical across decoding steps
• KV cache: store K and V for all past tokens per layer; on each new step only compute Q for the new token and attend over cached K/V
• Reduces per-step compute from O(n²·d) to O(n·d) — each new token costs time linear in context length, not quadratic
• Memory cost: 2 × seq_len × d_model × n_layers × sizeof(dtype); e.g., Llama 3 70B at 32K context ≈ 32GB in fp16
• KV cache is often the inference memory bottleneck, motivating GQA, MQA, and MLA for compression`
  },
  {
    id: "attn-4",
    theme: "Attention & Efficiency",
    question: "Explain MQA (Multi-Query Attention) and GQA (Grouped Query Attention).",
    answer:
`• MHA baseline: h query heads, h key heads, h value heads — full KV cache scales with h
• MQA (Shazeer 2019): single K and V head shared across all h query heads; KV cache is h× smaller; faster inference at a slight quality cost
• GQA (Ainslie et al. 2023): G groups of query heads share one K/V head each; interpolates between MHA (G=h) and MQA (G=1); balances quality and efficiency
• Used in: Llama 2 70B (GQA), Llama 3 all sizes, Mistral, Gemma 2/3
• Typical setting: 8 KV heads for a model with 32–64 query heads → 4–8× KV cache reduction`
  },
  {
    id: "attn-5",
    theme: "Attention & Efficiency",
    question: "What is attention logit soft-capping and why was it introduced in Gemma 2?",
    answer:
`• Problem: during large-scale training, attention logits (QKᵀ / √dₖ) can grow unboundedly large, pushing softmax toward a one-hot distribution; this causes training instability and loss spikes
• Soft-capping: applies a tanh-based bound to logits before softmax: score = softcap · tanh(QKᵀ / (√dₖ · softcap)); constrains scores to (−softcap, +softcap)
• Effect: limits the maximum influence any single token can exert; acts as implicit regularization on attention weights
• Benefits: more stable training loss curves, prevents attention collapse onto single tokens, better behavior on long-context tasks
• Used in Gemma 2; later architectural choices (p-RoPE, QK-Clip in MuonClip) address the same root cause in different ways`
  },
  {
    id: "attn-6",
    theme: "Attention & Efficiency",
    question: "What is Sliding Window Attention (SWA) and how does it reduce the cost of long-context inference?",
    answer:
`• Mechanism: each token attends only to the w preceding tokens (the "window") rather than the full sequence; KV cache stores only the last w key/value pairs per local layer
• Complexity: reduces per-token attention compute and KV memory from O(n) to O(w), where w ≪ n for long sequences
• Information propagation: through stacked layers the effective receptive field grows — a token can indirectly influence tokens up to w · n_layers positions away, allowing long-range dependencies without full attention at each layer
• Used in: Mistral (SWA with sliding KV cache), Gemma 3/4 (local attention layers in 5:1 local/global ratio), Llama 4 Scout (chunked attention in RoPE layers)
• Tradeoff: tokens outside the window are completely invisible within a single local layer — global coherence requires periodic full-attention layers in a hybrid design`
  },

  // ── Optimization ──────────────────────────────────────────────────────────
  {
    id: "opt-1",
    theme: "Optimization",
    question: "Compare Adam vs AdamW. Why does decoupled weight decay matter?",
    answer:
`• Adam: applies L2 regularization by adding λθ to the gradient before the adaptive update; effective weight decay scales as λ/√v̂ — larger for parameters with small gradient variance; inconsistent regularization across parameters
• AdamW (Loshchilov & Hutter 2019): decouples weight decay — applies θ ← θ − α·λθ directly after the gradient step, independent of adaptive scaling
• Effect: AdamW gives uniform regularization strength to all parameters regardless of their gradient history; more principled and consistent
• In practice: AdamW consistently outperforms Adam for Transformers and is the default optimizer for all major LLM training runs`
  },
  {
    id: "opt-2",
    theme: "Optimization",
    question: "What is gradient clipping and why is it used in LLM training?",
    answer:
`• Global norm clipping: compute L2 norm of all concatenated gradients; if ||g|| > threshold T, rescale: g ← g · (T / ||g||)
• Typical threshold: T = 1.0
• Prevents exploding gradients — common during periods of large curvature in the loss landscape, especially in early training
• Does NOT fix vanishing gradients — clipping only reduces magnitude, never amplifies
• LLMs trained without it are prone to occasional loss spikes; standard safety net even when Pre-LN + residuals handle most instability`
  },
  {
    id: "opt-3",
    theme: "Optimization",
    question: "Explain mixed-precision training (fp16/bf16). What is a loss scaler and when is it needed?",
    answer:
`• Keep master weights in fp32; compute forward and backward passes in fp16 or bf16
• bf16 vs fp16: both 16-bit total; bf16 has 8 exponent bits (same as fp32) vs fp16's 5 → bf16 has much larger dynamic range; bf16 is preferred for training
• fp16 underflow: gradients can round to zero due to limited dynamic range; requires loss scaling — multiply loss by large constant S, divide gradients by S after backward; skip update if inf/nan
• bf16 on A100/H100: large enough dynamic range that loss scaling is not needed
• FP8 training (H100): emerging standard delivering ~2× throughput over bf16; requires careful per-tensor scaling; first large-scale use in DeepSeek-V3`
  },
  {
    id: "opt-4",
    theme: "Optimization",
    question: "Compare SGD with momentum vs. Adam. When would you prefer one over the other?",
    answer:
`• SGD + momentum: accumulates exponential moving average of gradients (v ← βv + g); requires careful LR tuning; generalizes well with long training
• Adam: maintains per-parameter first moment m̂ and second moment v̂; effective LR = α / (√v̂ + ε); adapts to gradient magnitude; converges fast with little tuning
• Prefer SGD+momentum: image classifiers (ResNet), when generalization gap matters, long training runs with carefully tuned LR schedule
• Prefer Adam/AdamW: Transformers, NLP, fast prototyping, sparse gradients (embeddings)
• Key insight: Adam's fast convergence can lead to worse generalization ("sharp minima") — AdamW + cosine schedule partially mitigates this`
  },
  {
    id: "opt-5",
    theme: "Optimization",
    question: "What are common learning rate schedules for LLMs? Why do Transformers need warmup?",
    answer:
`• Linear warmup: LR increases from ~0 to peak over first N steps (typically 1–4% of total steps)
• Cosine decay: after warmup, LR follows cosine curve down to ~0 or a small floor
• Why warmup: at init, Adam's second moment v̂ is near zero → effective LR is very large → unstable updates in early steps; warmup prevents this
• Without warmup: early gradient steps are noisy and large → loss spikes or divergence
• Common schedule: warmup 2K steps → cosine decay to 10% of peak LR
• WSD (Warmup-Stable-Decay) used in Llama 3: stable phase with constant LR, then sharp final cosine decay`
  },

  // ── Evaluation ───────────────────────────────────────────────────────────
  {
    id: "eval-1",
    theme: "Evaluation",
    question: "What is perplexity? What does it measure and what are its limitations?",
    answer:
`• PPL = exp(−(1/n) Σ log p(wᵢ | w₁…wᵢ₋₁)) = exp(cross-entropy loss on held-out text)
• Interpretation: geometric mean inverse probability per token — lower is better; PPL=10 means the model assigns ~10% probability on average to the correct next token
• Limitations:
  - Tokenization-dependent: cannot compare PPL across models with different tokenizers
  - Distribution-dependent: must be evaluated on the same domain as training to be meaningful
  - Doesn't measure instruction-following, factuality, or helpfulness — a fine-tuned model may have higher PPL than the base model yet be far more useful`
  },
  {
    id: "eval-2",
    theme: "Evaluation",
    question: "What do MMLU, HellaSwag, and HumanEval measure? What are their blind spots?",
    answer:
`• MMLU (57 academic subjects, multiple-choice): tests world knowledge and reasoning breadth; blind spots: format-sensitive (MCQ ≠ open generation), high contamination risk when benchmark data appears in pretraining corpora
• HellaSwag (commonsense sentence completion): designed adversarially to fool models; largely saturated by modern LLMs — no longer a strong differentiator
• HumanEval (164 Python problems): measures pass@k — fraction of problems solved within k samples; blind spots: narrow domain (algorithmic Python), doesn't reflect real-world software engineering; SWE-bench is more realistic
• General blind spots: benchmark contamination in pretraining data; saturation; don't measure alignment, safety, long-horizon planning, or real-world helpfulness`
  },

  // ── Inference & Decoding ──────────────────────────────────────────────────
  {
    id: "infer-1",
    theme: "Inference & Decoding",
    question: "Compare greedy decoding, beam search, top-k sampling, and nucleus (top-p) sampling.",
    answer:
`• Greedy: argmax at each step; fast and deterministic; prone to repetition and degenerate loops
• Beam search: maintain top-b partial sequences at each step; better for constrained tasks (translation, summarization); produces generic/safe outputs for open-ended generation
• Top-k sampling: sample from the top k tokens by probability; fixed k is brittle — too narrow for peaked distributions, too broad for flat ones
• Nucleus (top-p) sampling: sample from the smallest token set whose cumulative probability ≥ p; adapts to sharpness of the distribution; typical p = 0.9–0.95
• Temperature: divide logits by T before softmax; T > 1 flattens distribution (more diverse), T < 1 sharpens it (more deterministic); T → 0 approaches greedy`
  },
  {
    id: "infer-2",
    theme: "Inference & Decoding",
    question: "What is speculative decoding and how does it speed up inference without quality loss?",
    answer:
`• Problem: large model inference is memory-bandwidth bound at batch size 1 — a small draft model runs 3–10× faster
• Algorithm:
  1. Draft model autoregressively generates k candidate tokens (cheap, fast)
  2. Large target model scores all k+1 positions in a single parallel forward pass
  3. Accept tokens left-to-right using rejection sampling: keep each accepted token, resample from the target at the first mismatch
• Correctness guarantee: produces the exact same output distribution as running the target model alone
• Speedup: 2–3× typical; requires draft and target to share tokenizer and produce similar output distributions
• Used in: Llama inference (smaller Llama as draft), DeepSeek-V3 inference (MTP head serves as the draft)`
  },

  // ── ML Fundamentals ───────────────────────────────────────────────────────
  {
    id: "ml-1",
    theme: "ML Fundamentals",
    question: "Explain the bias-variance tradeoff. How does it manifest in neural networks?",
    answer:
`• Bias: error from wrong model assumptions — model too simple to capture true pattern (underfitting)
• Variance: error from sensitivity to training data fluctuations — model memorizes noise (overfitting)
• Decomposition: Total error ≈ Bias² + Variance + Irreducible noise
• Neural networks: larger/deeper models → lower bias, higher variance → require more data and regularization
• Double descent: overparameterized NNs can interpolate training data (zero train loss) yet still generalize — the classic tradeoff breaks down at very large model/data scale
• Regularization tools: weight decay (L2), dropout, early stopping, data augmentation, LoRA`
  },
  {
    id: "ml-2",
    theme: "ML Fundamentals",
    question: "What is batch normalization? What problem does it solve and what are its drawbacks?",
    answer:
`• Normalizes activations to zero mean and unit variance per mini-batch per feature, then applies learned scale γ and shift β
• Solves: internal covariate shift (distributions shifting between layers slow training); enables higher learning rates; mild regularization effect
• Drawbacks: behavior differs between train (uses batch stats) and inference (uses running stats); degrades with small batch sizes; fails for variable-length sequences where padding distorts batch statistics
• LayerNorm (used in all Transformers): normalizes across the feature dimension per individual sample — no batch size dependence, identical behavior at train and inference, works naturally with sequences`
  },
  {
    id: "ml-3",
    theme: "ML Fundamentals",
    question: "What is the vanishing gradient problem and how do residual connections solve it?",
    answer:
`• In deep networks, gradients are products of Jacobians across many layers; if typical values are < 1, the product shrinks exponentially → gradients vanish near the input
• Residual connection: output y = F(x) + x; the identity shortcut carries gradient directly: ∂L/∂x = ∂L/∂y · (∂F/∂x + I)
• The identity term I ensures the gradient is always at least as large as ∂L/∂y — it cannot be killed by a single layer's Jacobian
• Enables training of very deep networks (ResNet, Transformer) that would otherwise fail to converge
• LSTM/GRU gates address the same problem in sequential RNNs via learned gating that controls gradient flow`
  },
  {
    id: "grad-1",
    theme: "ML Fundamentals",
    question: "What causes exploding gradients and how is it fixed beyond gradient clipping?",
    answer:
`• Causes: deep networks without residuals; large weight magnitudes; high LR; poor initialization → gradient product through layers grows exponentially
• Fixes beyond clipping:
  - Residual connections: gradient highway through identity shortcut (always has +1 term)
  - LayerNorm / BatchNorm: normalizes activations → bounds gradient magnitudes
  - Careful init: Xavier/Glorot (σ² = 2/(fan_in + fan_out)) for tanh; He (σ² = 2/fan_in) for ReLU — keeps variance ~1 through layers
  - Lower LR + warmup
  - Gradient clipping (global norm): rescales g if ||g|| > T
• In Transformers: Pre-LN + residual connections + AdamW largely solves this; clipping (T=1.0) added as safety net`
  },
  {
    id: "grad-2",
    theme: "ML Fundamentals",
    question: "Compare Xavier and He initialization. When do you use each?",
    answer:
`• Xavier/Glorot: W ~ U[−√(6/(fan_in+fan_out)), +√(6/(fan_in+fan_out))]
  → keeps variance ≈ 1 for linear/tanh activations where f'(0) ≈ 1
• He initialization: W ~ N(0, √(2/fan_in))
  → accounts for ReLU zeroing half its inputs; variance ≈ 1 after ReLU
• Rule: tanh/sigmoid → Xavier; ReLU/GeLU → He
• Transformers: typically use small normal init (σ=0.02) for embeddings and attention weights; some implementations scale residual branch output by 1/√(2·n_layers) to keep residual stream variance stable at depth`
  },

  // ── Alignment & RLHF ─────────────────────────────────────────────────────
  {
    id: "align-1",
    theme: "Alignment & RLHF",
    question: "What is RLAIF and how does it differ from RLHF?",
    answer:
`• RLAIF (Constitutional AI / Bai et al. 2022): replace human preference labelers with a strong LLM judge
• Same pipeline as RLHF: LLM scores (prompt, response A, response B) → reward model → PPO
• Advantages: scales without human annotation bottleneck; more consistent labeling; cheaper
• Disadvantages: inherits biases of the judge model; can reward verbose/confident-sounding outputs (verbosity bias)
• Used in production at Anthropic (Claude) and implicitly in many RLHF pipelines via LLM-as-judge`
  },
  {
    id: "align-2",
    theme: "Alignment & RLHF",
    question: "What are the key tradeoffs between RLHF+PPO, DPO, and RLAIF?",
    answer:
`• RLHF+PPO: most flexible; supports online data collection; most complex (separate RM, 4 models in memory); reward hacking risk
• DPO: eliminates RM and RL loop; trains on offline preference triples; simpler and more stable; no online exploration; slightly less peak quality
• RLAIF: scales annotation without humans; cheaper; inherits judge-model biases; verbosity/sycophancy risk
• In practice: DPO is preferred for compute-constrained settings; RLHF+PPO for frontier models where quality ceiling matters; RLAIF for scale`
  },

  // ── Regularization ────────────────────────────────────────────────────────
  {
    id: "reg-1",
    theme: "Regularization",
    question: "Compare L1 and L2 regularization. What are their effects on weights?",
    answer:
`• L2 (weight decay): adds λ||w||² to loss → gradient penalty ∝ w → shrinks all weights proportionally → dense solutions; well-suited for Transformers (AdamW)
• L1 (Lasso): adds λ||w||₁ → gradient penalty = λ·sign(w) → constant push toward zero regardless of magnitude → induces sparsity; some weights go exactly to zero
• L1 use cases: feature selection, sparse models
• L2 use cases: general regularization in deep learning
• Elastic Net: combination of L1 + L2
• In practice for LLMs: L2 via AdamW weight decay; L1 rarely used directly`
  },
  {
    id: "reg-2",
    theme: "Regularization",
    question: "How does dropout work? Where is it used in Transformers and why not during inference?",
    answer:
`• Randomly zero out each activation with probability p during training; scale remaining by 1/(1-p) (inverted dropout)
• Effect: prevents co-adaptation of neurons; acts as ensemble of exponentially many subnetworks
• In Transformers: applied after attention weights, after FFN, after residual addition (varies by implementation)
• Typical p = 0.1 for LLMs (lower than vision models — large LLMs don't need heavy dropout)
• Inference: disabled (p=0) — all units active; predictions are deterministic and use full capacity
• Note: dropout hurts performance when data is very large (GPT-3 trained without dropout)`
  },
  {
    id: "reg-3",
    theme: "Regularization",
    question: "What is label smoothing and when is it useful?",
    answer:
`• Instead of hard targets (0 or 1), use soft targets: y_smooth = (1-ε)·y_hard + ε/K where K = vocab size
• Prevents the model from becoming overconfident; regularizes the output distribution; improves calibration
• Effect: slight perplexity increase (since PPL assumes hard targets) but often better downstream accuracy and BLEU
• Used in: original Transformer (ε=0.1), machine translation, classification tasks
• Tradeoff: hurts perplexity metric but improves generalization`
  },
  {
    id: "reg-4",
    theme: "Regularization",
    question: "What is catastrophic forgetting and how is it mitigated during fine-tuning?",
    answer:
`• When fine-tuning on a new task, the model overwrites weights that encoded pretrained knowledge → performance on original tasks degrades
• Mitigations:
  - LoRA / PEFT: freeze base weights; only train low-rank adapters → pretrained knowledge preserved by construction
  - Low LR: smaller updates → less overwriting of pretrained representations
  - Replay / data mixing: include samples from pretraining distribution in fine-tuning data
  - EWC (Elastic Weight Consolidation): penalize changing weights that were important for the previous task (estimated via Fisher information)
  - Prompt tuning / prefix tuning: freeze all weights, only train soft prompt tokens
• For LLMs in practice: LoRA + data mixing is the standard solution`
  },

  // ── Normalization ─────────────────────────────────────────────────────────
  {
    id: "norm-1",
    theme: "Normalization",
    question: "Why do Transformers use LayerNorm instead of BatchNorm?",
    answer:
`• BatchNorm normalizes across the batch dimension (per feature): requires large batches; different stats at train vs inference; fails for variable-length sequences where padding distorts batch statistics
• LayerNorm normalizes across the feature dimension (per sample): stats computed per token independently; no batch size dependence; same behavior train and inference
• For sequences: each token position is one sample → LayerNorm is the natural fit
• Pre-LN vs Post-LN:
  - Post-LN (original Transformer): LN after residual; unstable without careful warmup
  - Pre-LN (GPT-2, Llama): LN before sublayer, inside residual branch; more stable training; preferred in modern LLMs
• RMSNorm (Llama 2/3, Gemma): simplified LN without mean centering — only scales by RMS; slightly faster; empirically matches LN quality`
  },

  // ── MoE Architecture ──────────────────────────────────────────────────────
  {
    id: "arch-1",
    theme: "Architecture: Dense vs. MoE",
    question: "Compare Dense models to Mixture-of-Experts (MoE) architectures. What are the primary trade-offs?",
    answer:
`• Dense model: every parameter is active for every token
  - Pros: simple, easier to train, high stability
  - Cons: computational cost (FLOPs) scales linearly with parameter count
• MoE model: a learned router selects a subset of expert sub-networks to process each token
  - Pros: massive effective capacity with much lower active compute; a 671B parameter model can run at the cost of a ~37B dense model (DeepSeek-V3)
  - Cons:
    - Memory bottleneck: all experts must reside in VRAM even though most are inactive per token
    - Routing instability: training the router risks 'expert collapse' where one expert handles all load
    - Communication overhead: in distributed training, moving tokens between experts on different GPUs is a major bottleneck`
  },
  {
    id: "moe-1",
    theme: "MoE Architecture",
    question: "What is Mixture-of-Experts (MoE) and how does it differ from a dense model?",
    answer:
`• Dense model: every parameter activated for every token — compute scales linearly with total params
• MoE: FFN layers replaced by N expert sub-networks + a learned router; only top-K experts activated per token
• Benefit: total capacity (parameters, stored knowledge) grows without proportional compute cost
• Sparsity ratio = N / K — e.g. DeepSeek-V3 has 256 routed experts, activates 8 → sparsity of 32
• Key challenge: load balancing — router must spread tokens across experts to avoid collapse onto a few
• Used in: DeepSeek-V3/R1, Kimi K2/K2.5, Llama 4 Maverick/Scout, Gemma 4 26B A4B, Mixtral`
  },
  {
    id: "moe-2",
    theme: "MoE Architecture",
    question: "What is routing collapse in MoE models and how is it solved?",
    answer:
`• Routing collapse: the router converges to always selecting the same 1–2 experts; other experts receive no gradient and become useless
• Traditional fix: auxiliary load-balancing loss — penalize imbalanced routing; but this degrades model performance by fighting the main objective
• DeepSeek-V3 innovation: auxiliary-loss-free load balancing — adds a learnable bias term to expert affinity scores during routing; bias is adjusted dynamically to equalize load without touching the main loss
• DeepSeekMoE also uses shared experts: 1 expert always activated per token to capture common knowledge, freeing routed experts to specialize`
  },

  // ── DeepSeek V3 / R1 ──────────────────────────────────────────────────────
  {
    id: "ds-1",
    theme: "DeepSeek V3 / R1",
    question: "What are the two core architectural innovations in DeepSeek-V3?",
    answer:
`1. Multi-head Latent Attention (MLA): compresses K and V tensors into a low-dimensional latent vector before caching; at inference, projects back to full size; reduces KV cache memory ~10× vs standard MHA while preserving model quality
2. DeepSeekMoE: ultra-fine-grained expert decomposition — 256 routed + 1 shared expert; only 8 routed experts activated per token; auxiliary-loss-free load balancing via affinity score bias terms
• Also: FP8 mixed-precision training (first at this scale), multi-token prediction (MTP) objective for both training signal and speculative decoding at inference`
  },
  {
    id: "ds-2",
    theme: "DeepSeek V3 / R1",
    question: "What is Multi-head Latent Attention (MLA) and why does it matter for inference?",
    answer:
`• Standard MHA: stores full K and V vectors per token per head per layer in KV cache → memory grows as O(n · d · n_heads · n_layers)
• MLA (DeepSeek-V2/V3/R1): projects K and V down to a compressed latent vector c (much smaller dimension) before caching; at inference, c is projected back up for attention computation
• Effect: KV cache footprint reduced ~10×, enabling much longer effective context and higher batch sizes at the same memory budget
• Tradeoff: extra projection compute per step, but memory bandwidth savings dominate at inference
• All DeepSeek V2, V3, R1 share identical MLA + MoE architecture; R1 differs only in training recipe (RLVR)`
  },
  {
    id: "ds-3",
    theme: "DeepSeek V3 / R1",
    question: "How was DeepSeek-R1 trained? What is RLVR and how does it differ from standard RLHF?",
    answer:
`• DeepSeek-R1-Zero: start from V3 base, apply large-scale RL with no SFT — uses GRPO (Group Relative Policy Optimization), a variant of PPO without a critic network
• RLVR (RL with Verifiable Rewards): reward signal comes entirely from rule-based verification (compiler, calculator, symbolic checker) rather than a learned reward model; avoids reward hacking since ground truth is unambiguous for math and code
• DeepSeek-R1: improves on R1-Zero by adding a cold-start SFT phase with high-quality CoT examples before RL, then SFT on both reasoning and non-reasoning data before final RL — fixes language mixing and readability issues present in R1-Zero
• DeepSeek also distilled R1 reasoning capabilities back into V3 via knowledge distillation, improving V3's reasoning without explicit RL`
  },
  {
    id: "ds-4",
    theme: "DeepSeek V3 / R1",
    question: "DeepSeek-V3 was trained on 14.8T tokens for ~$5.6M. What engineering choices made this possible?",
    answer:
`• MoE sparse activation: only 37B of 671B parameters activated per token → compute cost of a ~37B dense model
• FP8 mixed-precision training: first large-scale use at this scale; ~2× memory and compute savings vs bf16 with maintained quality
• Efficient pipeline parallelism across H800 clusters with custom multi-plane fat-tree network topology to minimize inter-node communication overhead
• Auxiliary-loss-free load balancing: no quality tax from balancing constraints
• MTP (multi-token prediction): training signal augmentation without extra inference cost
• Result: 180K H800 GPU-hours per trillion tokens — significantly cheaper than comparable dense models`
  },

  // ── Gemma 3 ───────────────────────────────────────────────────────────────
  {
    id: "gemma3-1",
    theme: "Gemma 3",
    question: "What is the local/global attention pattern in Gemma 3 and why was it introduced?",
    answer:
`• Problem: full global attention at 128K context → KV cache grows O(n) per layer → memory explosion
• Gemma 3 solution: interleave local and global attention layers in a 5:1 ratio (5 local layers per 1 global layer)
• Local attention layers: attend only to a short sliding window (~1024 tokens); KV cache stays small regardless of total sequence length
• Global attention layers: attend to full context; needed for coherence and long-range dependencies
• RoPE base frequency: local layers use 10K (same as Gemma 2), global layers use 1M — enables extrapolation to 128K tokens in global layers
• Net result: global capacity preserved while per-layer KV cache cost is dominated by the cheap local layers`
  },
  {
    id: "gemma3-2",
    theme: "Gemma 3",
    question: "How does Gemma 3 handle vision inputs? What is the Pan & Scan method?",
    answer:
`• Vision encoder: frozen SigLIP encoder (derived from PaliGemma); converts images to 256 soft token embeddings fed into the LM
• Problem: fixed 896×896 resolution loses fine detail in high-res or non-square images
• Pan & Scan (P&S): at inference, segment the image into overlapping crops adaptively based on aspect ratio and resolution; resize each crop to 896×896; encode separately; concatenate token sequences up to a maximum crop count
• Effect: effectively increases visual resolution for detailed tasks (reading small text, diagrams) at the cost of more tokens and compute
• Gemma 3 supports text + image; Gemma 3n (MatFormer) additionally supports audio`
  },
  {
    id: "gemma3-3",
    theme: "Gemma 3",
    question: "What is knowledge distillation and how was it used in Gemma 3's training?",
    answer:
`• Knowledge distillation: train a student model to match the soft output probability distributions of a larger teacher model, rather than hard one-hot labels; student learns to replicate the teacher's uncertainty and generalization
• Gemma 3 uses distillation from larger Gemini models: smaller variants (e.g., 4B) trained on softened teacher logits → Gemma 3 4B-IT is competitive with Gemma 2 27B-IT
• Combined with large token budgets: 14T tokens for the 27B model, 2T for the 1B model
• Advantage: smaller models punch above their weight class; allows Google to maintain a family of efficient edge-deployable models with near-frontier quality`
  },

  // ── Gemma 4 ───────────────────────────────────────────────────────────────
  {
    id: "gemma4-1",
    theme: "Gemma 4",
    question: "What are the three distinct model families in Gemma 4 and what are their design targets?",
    answer:
`• E2B / E4B (MatFormer / Gemma 3n): "E" = effective parameters; Matryoshka Transformer — nested sub-models within a single model; E4B contains E2B; can select intermediate sizes between 2B and 4B; Per-Layer Embedding (PLE) caching reduces runtime memory; targets mobile, edge, browser (Pixel, Chrome)
• 31B dense: standard decoder-only dense model; bridges gap between on-device and server-grade; targets local execution on workstations
• 26B A4B MoE: 26B total, 4B active per token (sparsity ~6.5); targets high-throughput inference; all 26B must be in memory for routing despite only 4B activating
• All Gemma 4 models: configurable thinking modes (reasoning on/off), extended multimodality (text + image; E2B/E4B also support video and audio natively)`
  },
  {
    id: "gemma4-2",
    theme: "Gemma 4",
    question: "What is the MatFormer / Matryoshka Transformer architecture used in Gemma 3n / Gemma 4 edge models?",
    answer:
`• MatFormer: nested architecture where a larger model contains a smaller model within its weights; named after Matryoshka (Russian nesting dolls)
• E4B contains E2B: the inner sub-model can be run independently at lower cost without loading outer parameters
• Per-Layer Embedding (PLE) caching: PLE parameters are a large fraction of total params but can be cached to fast local storage (NVMe, RAM) rather than loaded into GPU memory — reduces effective VRAM footprint substantially
• Conditional parameter loading: vision and audio parameter groups can be skipped entirely if the task is text-only — further reducing memory
• E2B with PLE caching + parameter skipping: >5B actual params but only ~1.91B effective parameters in VRAM
• Design target: full LLM capability at smartphone / edge device memory constraints`
  },
  {
    id: "gemma4-3",
    theme: "Gemma 4",
    question: "What is p-RoPE (proportional RoPE) and what is the theoretical motivation behind it?",
    answer:
`• Research finding (Oxford/DeepMind, ICLR 2025): in standard RoPE, the lowest-frequency dimension pairs are used by the model as near-position-independent semantic channels — the rotation angle changes so slowly at low frequencies that relative distance barely affects the dot product; but these channels break at long context because the near-constant behavior fails as distances grow beyond training length
• p-RoPE fix: zero out the rotations for the lowest (1-p) fraction of frequency pairs, making them explicitly position-independent (NoPE channels); only the top p fraction of frequencies carry positional signal
• Gemma 4 uses p=0.25: only 25% of RoPE pairs rotate (positional), 75% are pure semantic channels with no rotation → those channels are robust by construction regardless of context length
• Empirical result: p-RoPE with p=0.5 improves validation perplexity on 2B models over standard RoPE; p=0.25 still significantly outperforms NoPE
• Applied selectively: Gemma 4 uses p-RoPE only on global full-attention layers (theta=1M); local sliding-window layers retain standard RoPE (theta=10K) where short-range position precision matters more`
  },
  {
    id: "gemma4-4",
    theme: "Gemma 4",
    question: "What is K=V weight sharing in Gemma 4's global attention layers and what does it trade off?",
    answer:
`• Standard MHA/GQA: separate W_K and W_V projection matrices; keys and values are independent tensors
• Gemma 4 global layers: W_V is eliminated entirely — the key tensor is reused directly as the value tensor (K=V); no separate value projection
• Combined with fewer KV heads in global layers (4 heads vs 16 in local layers) and wider head_dim (512 vs 256): global layers carry far fewer KV parameters
• KV cache impact: each global layer stores only one set of tensors instead of two, halving the per-layer cache footprint for those layers
• Tradeoff: removing W_V means values cannot learn a different representation from keys; the model must encode "what to attend to" and "what to extract" in the same tensor — a capacity constraint accepted in exchange for parameter and memory efficiency
• Complemented by V-norm: RMSNorm applied to the shared K=V values without learned scale, stabilizing activations without adding parameters`
  },
  {
    id: "gemma4-5",
    theme: "Gemma 4",
    question: "How does Gemma 4's dual attention geometry differ from Gemma 3, and why does it matter?",
    answer:
`• Gemma 3: identical attention geometry across all layers — same head_dim, same number of KV heads regardless of whether the layer is local or global
• Gemma 4 splits: local sliding-window layers (head_dim=256, 16 KV heads, standard RoPE theta=10K) vs global full-context layers (head_dim=512, 4 KV heads, p-RoPE theta=1M, K=V sharing)
• Design rationale:
  - Local layers need fine-grained position discrimination within a 512–1024 token window → smaller heads, more heads, high-frequency standard RoPE
  - Global layers need robust long-range semantic attention across 256K tokens → wider heads for richer representations, fewer KV heads (less memory), p-RoPE for context-length robustness, K=V to cut parameters
• Net result: "same attention, different window" (Gemma 3) → "different attention, different window" (Gemma 4)
• The final layer is always global, ensuring the output representation integrates full-context information`
  },
  {
    id: "gemma4-6",
    theme: "Gemma 4",
    question: "What is shared KV cache in Gemma 4 and how does it differ from standard per-layer KV caching?",
    answer:
`• Standard KV cache: every layer independently computes and stores its own K and V tensors; memory = n_layers × seq_len × n_kv_heads × head_dim × 2
• Gemma 4 shared KV cache (edge models): the last N layers reuse the KV tensors computed by an earlier layer rather than computing new projections; those layers have no W_K/W_V of their own
• Effect: dramatically cuts cache memory — the edge models (E2B/E4B) achieve ~83% reduction in KV cache compared to per-layer caching, critical for fitting 256K context on device
• Cost: layers sharing a KV cache cannot specialize their key/value representations; they all attend to the same pre-computed context — a representational constraint acceptable for later layers that do more integration than feature extraction
• This is an edge-model technique; the 31B and 26B server models use per-layer caches with K=V weight sharing and fewer global KV heads to reduce cache size instead`
  },

  // ── Kimi K2 / K2.5 ───────────────────────────────────────────────────────
  {
    id: "kimi-1",
    theme: "Kimi K2 / K2.5",
    question: "What are the key architectural and training differences between Kimi K2 and DeepSeek-V3?",
    answer:
`• Both use ultra-sparse MoE + MLA; K2 is explicitly modeled on DeepSeek-V3's architecture
• Key differences:
  - Experts: K2 has 384 routed experts vs V3's 256; sparsity = 48 (384/8) vs 32 (256/8) — higher sparsity yields better validation loss per K2's scaling law analysis
  - Attention heads: K2 uses 64 heads vs V3's 128; reduction trades marginal quality for lower inference overhead at long context (agentic use cases have long contexts)
  - Scale: K2 = 1.04T total / 32B active vs V3 = 671B total / 37B active
  - Optimizer: K2 uses MuonClip (Muon + QK-Clip) instead of AdamW — claimed to be more token-efficient
  - Focus: K2 explicitly optimized for agentic tasks (tool use, SWE-bench, multi-step planning) vs V3's general-purpose profile`
  },
  {
    id: "kimi-2",
    theme: "Kimi K2 / K2.5",
    question: "What is the MuonClip optimizer and what problem does it solve?",
    answer:
`• Muon optimizer: applies Newton-Schulz orthogonalization to gradient matrices before the update step — more token-efficient than Adam (more learning signal per token consumed)
• Problem at scale: Muon causes training instability in large MoE models — attention logits (QKᵀ scores) can explode, causing loss spikes
• QK-Clip: clips the maximum attention logit to a threshold τ (e.g., 100) during training; prevents attention score explosion; clip frequency naturally decays as training stabilizes
• MuonClip = Muon + QK-Clip: enabled stable pre-training of K2 on 15.5T tokens with zero loss spikes — a claimed first for trillion-parameter scale
• Significance: Adam's per-parameter adaptive LR is well-suited for sparse gradients but wastes signal on dense updates; Muon is more efficient for the dense weight matrices in transformer blocks`
  },
  {
    id: "kimi-3",
    theme: "Kimi K2 / K2.5",
    question: "What is K2.5's Agent Swarm mode and what are its tradeoffs?",
    answer:
`• Agent Swarm: K2.5 can decompose a complex task into parallel sub-tasks and dynamically instantiate domain-specific sub-agents that execute concurrently
• Up to 100 sub-agents in parallel; each is a separate model inference call
• Tradeoff: wall-clock latency drops dramatically vs sequential execution; total token consumption (and cost) scales with number of active agents — a 100-agent swarm may solve in seconds but hammers the inference API
• Requires careful application-level design: developer must balance latency vs compute expenditure
• K2.5 also adds: native multimodal pre-training (vision + text co-trained from scratch via 400M MoonViT-3D encoder), 256K context window (vs K2's 128K), and instant/thinking/agent/swarm operational modes`
  },

  // ── Llama 4 ───────────────────────────────────────────────────────────────
  {
    id: "llama4-1",
    theme: "Llama 4",
    question: "What are the key architectural differences between Llama 4 Scout and Maverick?",
    answer:
`• Both: first MoE generation of Llama; early fusion multimodality (text + image tokens unified in backbone); 17B active parameters per token; FP8 training; ~30–40T token pretraining
• Scout (17B active / 109B total): 16 experts; full MoE across all layers; 10M token context window via iRoPE; fits on single H100 with int4 quantization; optimized for efficient long-context deployment
• Maverick (17B active / 400B total): 128 experts; MoE and dense layers alternate (MoE in half the layers); 1M token context; co-distilled from Llama 4 Behemoth teacher model; higher absolute quality
• iRoPE (Scout): interleaves NoPE layers (no positional embedding, full causal mask, every 4th layer) with RoPE layers (chunked attention for efficiency); NoPE layers enable length generalization beyond training distribution`
  },
  {
    id: "llama4-2",
    theme: "Llama 4",
    question: "What is early fusion multimodality and how does it differ from the late-fusion approach used in earlier models?",
    answer:
`• Late fusion (e.g., LLaVA, Gemma 3): encode image separately with a frozen vision encoder → project to token embeddings → concatenate with text tokens and feed to LLM; vision encoder and LLM trained largely independently
• Early fusion (Llama 4, Kimi K2.5): vision and text tokens are unified in the model's core representation from the start of pretraining; the model learns joint representations rather than bridging two separately trained systems
• Benefits: better cross-modal reasoning; model can natively attend between visual and textual tokens at every layer; no modality gap from separate encoders
• Cost: more complex pretraining; vision data must be included from the very beginning of the training run
• Llama 4 early fusion allows processing text, image, and video within a single architecture`
  },
  {
    id: "llama4-3",
    theme: "Llama 4",
    question: "What is iRoPE and why does Llama 4 Scout need it for 10M token context?",
    answer:
`• Standard RoPE: assigns a rotation to each position based on its absolute index; extrapolation beyond training length degrades because the model has never seen those rotation angles
• iRoPE (interleaved RoPE): every 4th attention layer uses NoPE — no positional embedding at all, just a standard causal mask; the remaining 3 layers use RoPE with chunked local attention
• NoPE layers: without any positional bias, these layers can attend globally with no positional length limit; they serve as the long-range integration mechanism
• RoPE layers with chunked attention: provide local positional structure efficiently without quadratic cost over 10M tokens
• Combined with temperature scaling on attention logits: iRoPE allows Scout to generalize to sequences far beyond its 256K pretraining length, extending to 10M during instruction-tuning`
  },

  // ── Math & Statistics Foundations ────────────────────────────────────────
  {
    id: "math-1",
    theme: "ML Fundamentals",
    question: "What is the sigmoid function? Give the formula and describe its properties.",
    answer:
`• Formula: σ(x) = 1 / (1 + e^{−x})
• Output range: (0, 1) — maps any real number to a probability-like value
• Derivative: σ'(x) = σ(x) · (1 − σ(x)) — computable from the output itself without recomputing the input
• Shape: S-curve symmetric around (0, 0.5); σ(0) = 0.5; σ(x) → 1 as x → +∞; σ(x) → 0 as x → −∞
• Limitations: saturates for large |x| → near-zero gradients → vanishing gradient problem in deep networks; outputs not zero-centered → can slow training
• Use cases: binary classification output layer; gates in LSTMs/GRUs; logistic regression`
  },
  {
    id: "math-2",
    theme: "ML Fundamentals",
    question: "What is the difference between variance and standard deviation?",
    answer:
`• Variance: σ² = (1/n) Σ (xᵢ − μ)² — average squared deviation from the mean; units are squared relative to the data
• Standard deviation: σ = √variance — same units as the data; more interpretable as a measure of spread
• When to use variance: mathematical derivations (variance is additive for independent variables: Var(X+Y) = Var(X) + Var(Y)); used in optimization objectives (MSE)
• When to use std: communicating spread in original units; z-score normalization: z = (x − μ) / σ; reporting model error
• Sample vs population: sample variance divides by (n−1) rather than n (Bessel's correction) for unbiased estimation from a sample`
  },

  // ── Common Recipes (interview quick-reference) ────────────────────────────
  {
    id: "recipe-1",
    theme: "ML Fundamentals",
    question: "What is a Transformer? How do encoders and decoders differ?",
    answer:
`• Transformer (Vaswani et al. 2017): sequence model built entirely on self-attention; replaces recurrence (RNNs) and convolutions; every position can directly attend to every other in one step
• Core mechanism: Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V; multi-head variant runs h parallel attention functions then concatenates
• Encoder: bidirectional — each token attends to all tokens; produces rich contextual representations; used for understanding tasks (BERT, RoBERTa, embeddings)
• Decoder: causal/autoregressive — each token attends only to previous tokens (masked); generates one token at a time; used for generation (GPT, Llama, Gemma)
• Encoder-decoder (original Transformer): encoder processes input; decoder generates output via cross-attention over encoder states; natural for seq2seq (translation, summarization)
• Modern LLMs are decoder-only: simpler architecture, scales well, pretraining via next-token prediction`
  },
  {
    id: "recipe-2",
    theme: "ML Fundamentals",
    question: "Why do we use softmax for multiclass classification? What does the temperature parameter do?",
    answer:
`• Softmax: σ(z)ᵢ = e^{zᵢ} / Σⱼ e^{zⱼ} — converts raw logits into a valid probability distribution (all positive, sum to 1)
• Why not linear normalization? Softmax amplifies differences between logits exponentially — the largest logit dominates — making the model decisive while still preserving gradient signal for all classes
• Numerical stability: compute softmax(z − max(z)) to prevent exp overflow; mathematically equivalent
• Temperature T: replace z with z/T before softmax
  - T = 1: standard softmax
  - T > 1: flattens distribution → more uniform, more random (used in sampling, knowledge distillation to soften teacher outputs)
  - T < 1: sharpens distribution → more peaked, more confident
  - T → 0: approaches argmax (greedy decoding)
  - T → ∞: approaches uniform distribution`
  },
  {
    id: "recipe-3",
    theme: "ML Fundamentals",
    question: "What are common activation functions and when do you use each?",
    answer:
`• Sigmoid: σ(x) = 1/(1+e^{−x}); range (0,1); saturates at extremes → vanishing gradients; use for binary output layers and gates
• Tanh: range (−1,1); zero-centered (better than sigmoid for hidden layers); still saturates; use in RNNs
• ReLU: max(0,x); no saturation for x>0; fast; risk of "dying ReLU" (dead neurons when x<0 permanently); default for CNNs/MLPs
• Leaky ReLU: max(αx, x) with small α (e.g. 0.01); prevents dead neurons; slightly more expensive
• GeLU: x · Φ(x) where Φ is Gaussian CDF; smooth approximation of ReLU; default in BERT, GPT-family
• SiLU / Swish: x · σ(x); smooth, non-monotonic; used in EfficientNet and some LLMs
• GeGLU: (xW₁) ⊗ GeLU(xW₂); gated variant; standard FFN activation in Llama, Gemma
• Softmax: e^{zᵢ}/Σe^{zⱼ}; converts logits to probabilities; used only at output layer for classification`
  },
  {
    id: "recipe-4",
    theme: "ML Fundamentals",
    question: "What are common loss functions and when do you use each?",
    answer:
`• Cross-entropy (classification): L = −Σ yᵢ log ŷᵢ; binary: −[y log ŷ + (1−y) log(1−ŷ)]; penalizes confident wrong predictions heavily; standard for classification and LLM next-token prediction
• MSE (regression): L = (1/n) Σ (yᵢ − ŷᵢ)²; differentiable everywhere; sensitive to outliers (squaring amplifies large errors)
• MAE (regression): L = (1/n) Σ |yᵢ − ŷᵢ|; robust to outliers; non-differentiable at 0 (use subgradient)
• Huber loss: MSE for |y−ŷ| < δ, MAE-like otherwise; combines robustness and smooth differentiability; use when data has occasional large outliers
• KL Divergence: D_KL(P||Q) = Σ P(x) log(P(x)/Q(x)); measures how Q diverges from P; used in VAEs, knowledge distillation, RLHF KL penalty
• NLL (Negative Log-Likelihood): −log p(y|x); for LLMs identical to cross-entropy when targets are one-hot
• Bradley-Terry (pairwise preference): −log σ(r(y_w) − r(y_l)); used for reward model training in RLHF`
  },
  {
    id: "recipe-5",
    theme: "Optimization",
    question: "What are common optimizers? When do you prefer each?",
    answer:
`• SGD: w ← w − η·∇L; simple, well-understood; requires careful LR tuning; best generalization in vision with long schedules
• SGD + Momentum: v ← βv + ∇L; w ← w − η·v; smooths oscillations; faster than vanilla SGD
• Adam: maintains per-parameter first moment m̂ and second moment v̂; effective LR = α·m̂/(√v̂ + ε); fast convergence with little tuning; default for NLP/Transformers
• AdamW: decouples weight decay — applies λ·w directly after gradient step, not via gradient; uniform regularization; preferred over Adam for all Transformer training
• Adagrad: accumulates squared gradients; per-parameter LR shrinks over time; good for sparse gradients (word embeddings); LR can decay to zero — limited for deep learning
• RMSprop: exponential moving average of squared gradients (fixes Adagrad's decay); precursor to Adam
• Muon (emerging): Newton-Schulz orthogonalization of gradient matrices; more token-efficient than Adam for dense weight matrices; used in Kimi K2
• Rule of thumb: AdamW for Transformers; SGD+momentum for CNNs/vision when training budget allows`
  },
  {
    id: "recipe-6",
    theme: "Optimization",
    question: "What are common learning rate schedules?",
    answer:
`• Constant: fixed LR throughout; simple; rarely optimal for long runs
• Step decay: reduce LR by factor γ every N epochs; common in image classification
• Exponential decay: LR = LR₀ · γᵗ; smooth continuous reduction
• Cosine annealing: LR = LR_min + ½(LR_max − LR_min)(1 + cos(πt/T)); smooth decay to minimum; standard for LLM pretraining
• Linear warmup + cosine decay: ramp up from ~0 to peak over first 1–4% of steps then cosine decay; most common LLM schedule; warmup prevents Adam instability at init (v̂ ≈ 0 → huge effective LR)
• WSD (Warmup-Stable-Decay): warmup → long stable plateau at peak LR → sharp final cosine decay; used in Llama 3; allows mid-training checkpoints without quality degradation
• Cyclical LR: oscillates between min/max LR; helps escape local minima; useful for fine-tuning`
  },
  {
    id: "recipe-7",
    theme: "Regularization",
    question: "What is overfitting? How do you detect and prevent it?",
    answer:
`• Overfitting: model memorizes training data including noise → low train loss, high val/test loss; too much capacity relative to data size
• Detection: train loss decreasing while val loss increases or plateaus (diverging curves); large gap between train and val accuracy
• Prevention:
  - More data / data augmentation: most reliable fix
  - L2 regularization (weight decay): penalizes large weights; built into AdamW
  - Dropout: randomly zero activations during training; prevents co-adaptation
  - Early stopping: monitor val loss; stop when it stops improving for N steps
  - Reduce model capacity: fewer layers or smaller hidden dim
  - Label smoothing: softens target distribution; prevents overconfident predictions
  - For LLMs: LoRA (freeze base, train adapters), data mixing (add pretraining data to fine-tuning), low LR
• Modern insight: overparameterized models can still generalize when trained with weight decay + small LR + large data — "benign overfitting" / double descent`
  },
  {
    id: "recipe-8",
    theme: "ML Fundamentals",
    question: "What are vanishing and exploding gradients? How do you prevent each?",
    answer:
`• Vanishing gradients: during backprop, gradient magnitudes shrink exponentially through layers — weights near the input receive negligible updates; deep networks fail to train
  - Causes: sigmoid/tanh saturating activations (derivative → 0 at extremes); multiplicative chain rule in deep networks
  - Fixes: ReLU/GeLU (no saturation for x>0); residual connections (identity shortcut carries gradient unchanged); LayerNorm; Pre-LN placement; LSTM/GRU gates for RNNs
• Exploding gradients: gradient magnitudes grow exponentially → loss spikes, NaN, divergence
  - Causes: large weight magnitudes; high LR; deep networks without normalization or residuals
  - Fixes: gradient clipping (rescale g if ||g|| > T, typically T=1.0); Xavier/He initialization; lower LR + warmup; LayerNorm
• Standard Transformer recipe that prevents both: Pre-LayerNorm + residual connections + He/small-σ init + AdamW + gradient clipping (T=1.0)`
  },
  {
    id: "recipe-9",
    theme: "Optimization",
    question: "What do you do if training loss doesn't converge?",
    answer:
`1. Overfit one batch first: train on a single batch until loss → 0; if it can't, there's a bug in the model or data pipeline — fix this before anything else
2. Check learning rate: too high → loss oscillates or diverges; too low → barely decreasing; try reducing by 10× or sweep {1e-2, 1e-3, 1e-4}
3. Check for NaN/inf: add gradient/activation logging; common sources: log(0), division by zero, fp16 overflow (add loss scaling or switch to bf16)
4. Check data: NaN inputs? Label leakage? Wrong normalization? Class imbalance? Shuffle the dataset
5. Check loss function: correct for the task? Numerically stable? (prefer log-softmax + NLL over softmax + log to avoid log(near-zero))
6. Check initialization: weights too large → exploding activations; use Xavier (tanh) or He (ReLU/GeLU) init
7. Check gradient flow: log per-layer gradient norms; vanishing → residuals, better activations; exploding → clip or reduce LR
8. Reduce batch size: larger batches flatten the loss landscape and can miss narrow minima; smaller batches add noise that acts as regularization
9. Add warmup: without it, Adam's v̂ ≈ 0 at init → huge effective LR → early divergence`
  },

  // ── Frontier Model Comparison ─────────────────────────────────────────────
  {
    id: "frontier-1",
    theme: "Frontier Model Comparison",
    question: "Compare the MLA approach (DeepSeek/Kimi) vs GQA (Llama 3, Gemma 3) for KV cache reduction.",
    answer:
`• GQA (Grouped Query Attention): group Q heads share fewer K/V heads (G groups); reduces KV cache by factor of n_heads/G; still stores one K/V pair per token per group
• MLA (Multi-head Latent Attention): compresses K/V for all heads into a single low-rank latent vector c per token; at inference, project c back up; compression ratio is much higher (~10× vs MHA, vs ~4–8× for typical GQA)
• GQA advantage: simpler, no projection overhead at inference, compatible with standard attention kernels
• MLA advantage: far greater KV cache compression, critical for very long contexts or large batch sizes
• In practice: GQA is standard in most models (Llama 3, Gemma, Mistral); MLA is DeepSeek's differentiator, adopted by Kimi K2/K2.5 due to demonstrated efficiency gains`
  },
  {
    id: "frontier-2",
    theme: "Frontier Model Comparison",
    question: "What is the significance of the MoE trend across Llama 4, Kimi K2, DeepSeek-V3, and Gemma 4?",
    answer:
`• All four major open-weight model families released in 2024–2025 converged on MoE as the architecture of choice for frontier-scale models
• Core reason: MoE decouples model capacity (total params) from inference cost (active params) — same quality as a much larger dense model at a fraction of compute per token
• Sparsity varies: Gemma 4 26B A4B (sparsity ~6.5), Llama 4 Scout (sparsity ~6.4), DeepSeek-V3 (sparsity 32), Kimi K2 (sparsity 48) — higher sparsity = more capacity at same FLOPs, but harder load balancing
• Remaining challenge: MoE models require full parameter load in memory for routing even when most are inactive — memory cost is dense-model-like despite sparse compute
• Implication for DeepMind: Gemini 1.5/2.0 and Gemma 4 all use MoE; understanding MoE tradeoffs (load balancing, routing collapse, memory vs compute) is directly relevant`
  },
];

// Filter groups shown in the app. A theme not listed here lands in "Other".
const THEME_GROUPS = [
  { id: "foundations", label: "Foundations",
    themes: ["ML Fundamentals", "Optimization", "Regularization", "Normalization", "Evaluation"] },
  { id: "transformers", label: "Transformers & Attention",
    themes: ["Transformer Architecture", "Attention & Efficiency", "MoE Architecture",
             "Architecture: Dense vs. MoE", "Inference & Decoding"] },
  { id: "training", label: "Training & Alignment",
    themes: ["LLM Training Pipeline", "Alignment & RLHF"] },
  { id: "frontier", label: "Frontier Models",
    themes: ["DeepSeek V3 / R1", "Gemma 3", "Gemma 4", "Kimi K2 / K2.5", "Llama 4",
             "Frontier Model Comparison"] },
];
