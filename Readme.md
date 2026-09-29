# ML Interview Prep — Quiz App

### ▶ [Play it in your browser](https://deveaup.github.io/ml_interview_study/)

A flashcard quiz for ML engineer interview prep, with spaced-repetition-style
weighting so the questions you keep getting wrong show up more often.

74 questions across 18 themes: ML fundamentals, optimization, regularization,
the Transformer, attention efficiency, MoE, the LLM training pipeline,
alignment & RLHF, inference, evaluation, and recent frontier models
(Gemma 3 / 4, Llama 4, DeepSeek V3 / R1, Kimi K2 / K2.5).

Nothing to install and no account: it runs entirely in the browser.

## How to use it

1. Open the [live version](https://deveaup.github.io/ml_interview_study/), or
   clone the repo and open `index.html` in any browser (no server needed).
2. Read the question, then click **Show Answer** to reveal it.
3. Mark yourself honestly:
   - **✓ Got it** — you knew the answer.
   - **✗ Missed it** — you didn't, or only partially.
   - **Skip** — move on without scoring (doesn't affect priority).
4. Use the theme filter buttons at the top to focus on a specific topic
   (e.g. "MoE Architecture", "Attention & Efficiency") — click **All** to clear filters.
5. Your progress is saved automatically in the browser's local storage
   (per-browser, per-device — it won't sync across machines).
6. Click **reset** (top right) to wipe all progress and start over.

The header badges (`— high`, `— med`, `— low`) and the progress bar show how many
questions currently sit in each priority bucket.

## How the frequency system works

Each question tracks a **consecutive-correct counter (`cc`)** that goes up every
time you answer "Got it" in a row, and resets to `0` the moment you miss it.
That counter maps to a priority level, which controls how often the question
is picked:

| Priority | Consecutive correct | Sampling weight | Meaning |
|----------|---------------------|------------------|---------|
| 🔴 High   | 0 (never gotten right, or just missed) | **10x** | Shown most often — needs work |
| 🟠 Medium | 1–2 in a row | **3x** | Getting there — shown occasionally |
| 🟢 Low ("mastered") | 3+ in a row | **1x** | Rarely shown — you know it |

Questions are picked with **weighted random sampling** each round: a "high"
question is 10x more likely to appear than a "low" one on any given draw, but
nothing is ever fully excluded — even mastered questions resurface now and
then to keep the memory fresh. Missing a question once immediately drops it
back to "high" priority, no matter how many times you'd gotten it right before.

## Files

- `index.html` — the app (structure, styling, and all quiz logic in one file).
- `ml-quiz.html` — redirect to `index.html`, kept so old links still work.
- `questions.js` — the live question bank loaded by the app (`QUESTIONS` array), grouped by theme.
- `drafts/questions.json` — a draft/scratch copy of the question bank with `TODO` placeholder answers; not loaded by the app.
- `transformer_diagram.md` — supplementary notes/diagram on the Transformer architecture.

## Adding questions

Questions live in `questions.js` as plain objects:

```js
{
  id: "tf-7",                        // unique; prefix by theme
  theme: "Transformer Architecture", // becomes a filter button
  question: "…",
  answer: `…`                        // template string, shown as-is
},
```

Progress is keyed by `id`, so keep existing ids stable when editing.
