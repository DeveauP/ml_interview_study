# Transformer Architecture Diagram

```mermaid
flowchart TD
    %% ── INPUT ──────────────────────────────────────────────
    subgraph INPUT["INPUT PROCESSING"]
        direction TB
        TK["Token IDs"] --> EMB["Embedding Lookup\nd_model dims"]
        PE["Positional Encoding\n(sinusoidal / learned)"] --> ADD_IN(("+"))
        EMB --> ADD_IN
        ADD_IN --> X["x  ∈  ℝ^(seq × d_model)"]
    end

    %% ── ENCODER ────────────────────────────────────────────
    subgraph ENC["ENCODER BLOCK  ×N"]
        direction TB
        E_IN["x (from input / prev layer)"]
        E_IN --> MHSA["Multi-Head\nSelf-Attention\n─────────────\nbidirectional\nall→all"]
        MHSA --> E_ADD1(("+"))
        E_IN  --> E_ADD1
        E_ADD1 --> E_LN1["LayerNorm"]
        E_LN1 --> E_FFN["FFN\nLinear → GeLU/ReLU → Linear\n4·d_model hidden dim"]
        E_FFN --> E_ADD2(("+"))
        E_LN1 --> E_ADD2
        E_ADD2 --> E_LN2["LayerNorm"]
        E_LN2 --> ENC_OUT["Encoder Output K, V\n(passed to each decoder layer)"]
    end

    %% ── DECODER ────────────────────────────────────────────
    subgraph DEC["DECODER BLOCK  ×N"]
        direction TB
        D_IN["y (target tokens,\nshifted right)"]
        D_IN --> MMHSA["Masked MHA\n─────────────\ncausal / unidirectional\nfuture tokens masked"]
        MMHSA --> D_ADD1(("+"))
        D_IN   --> D_ADD1
        D_ADD1 --> D_LN1["LayerNorm"]

        D_LN1 --> CA["Cross-Attention\n─────────────\nQ  = decoder state\nK,V = encoder output"]
        CA --> D_ADD2(("+"))
        D_LN1 --> D_ADD2
        D_ADD2 --> D_LN2["LayerNorm"]

        D_LN2 --> D_FFN["FFN\nLinear → GeLU/ReLU → Linear"]
        D_FFN --> D_ADD3(("+"))
        D_LN2 --> D_ADD3
        D_ADD3 --> D_LN3["LayerNorm"]
    end

    %% ── ATTENTION DETAIL ───────────────────────────────────
    subgraph SDPA["SCALED DOT-PRODUCT ATTENTION"]
        direction LR
        Q_in["Q"] --> QK["Q · Kᵀ"]
        K_in["K"] --> QK
        QK --> SCALE["÷ √dₖ"]
        SCALE --> MASK["(+ mask\noptional)"]
        MASK --> SM["softmax"]
        SM --> AV["× V"]
        V_in["V"] --> AV
        AV --> A_out["Attention output"]
    end

    %% ── MULTI-HEAD ──────────────────────────────────────────
    subgraph MHA["MULTI-HEAD ATTENTION"]
        direction TB
        MH_IN["Input X"] --> WQ1["W_Q¹"] & WQ2["W_Q²"] & WQh["W_Qʰ"]
        MH_IN            --> WK1["W_K¹"] & WK2["W_K²"] & WKh["W_Kʰ"]
        MH_IN            --> WV1["W_V¹"] & WV2["W_V²"] & WVh["W_Vʰ"]
        WQ1 & WK1 & WV1 --> H1["head₁\nSDPA"]
        WQ2 & WK2 & WV2 --> H2["head₂\nSDPA"]
        WQh & WKh & WVh --> Hh["headₕ\nSDPA"]
        H1 & H2 & Hh --> CONCAT["Concat(head₁ … headₕ)"]
        CONCAT --> WO["× W_O"]
        WO --> MH_OUT["MultiHead output\n∈ ℝ^(seq × d_model)"]
    end

    %% ── OUTPUT ──────────────────────────────────────────────
    subgraph OUT["OUTPUT HEAD"]
        direction TB
        D_LN3 --> LIN["Linear\nd_model → vocab_size"]
        LIN --> SMAX["softmax\n→ P(next token | context)"]
    end

    %% ── FLOW CONNECTIONS ────────────────────────────────────
    INPUT  --> ENC
    INPUT  --> DEC
    ENC_OUT-->|"K, V injected\ninto every decoder layer"| CA

    %% ── STYLE ───────────────────────────────────────────────
    classDef block  fill:#1e3a5f,stroke:#4a9eff,color:#e8f4fd,rx:6
    classDef norm   fill:#2d4a2d,stroke:#66bb6a,color:#e8f5e9,rx:4
    classDef add    fill:#4a3800,stroke:#ffa000,color:#fff8e1,rx:50%
    classDef attn   fill:#3b1f5e,stroke:#ab47bc,color:#f3e5f5,rx:6
    classDef ffn    fill:#1a3a3a,stroke:#26c6da,color:#e0f7fa,rx:6
    classDef io     fill:#2a2a2a,stroke:#90a4ae,color:#eceff1,rx:4

    class MHSA,MMHSA,CA,H1,H2,Hh attn
    class E_FFN,D_FFN ffn
    class E_LN1,E_LN2,D_LN1,D_LN2,D_LN3 norm
    class E_ADD1,E_ADD2,D_ADD1,D_ADD2,D_ADD3 add
    class TK,PE,X,ENC_OUT,D_LN3,SMAX io
```

---

## Quick-reference formulas

| Component | Formula |
|---|---|
| Scaled dot-product attention | `Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V` |
| Multi-head | `MultiHead = Concat(head₁…headₕ) · W_O` |
| Per head projections | `W_Q, W_K, W_V ∈ ℝ^(d_model × dₖ)`,  `dₖ = d_model / h` |
| FFN | `FFN(x) = W₂ · GeLU(W₁x + b₁) + b₂` |
| Residual + norm pattern | `LayerNorm(x + Sublayer(x))` |

## Architecture variants

| Model type | Encoder | Decoder |
|---|---|---|
| **Encoder-only** (BERT) | ✅ bidirectional self-attn | ✗ |
| **Decoder-only** (GPT / modern LLMs) | ✗ | ✅ causal masked self-attn only |
| **Encoder–Decoder** (T5, original paper) | ✅ | ✅ + cross-attention |
