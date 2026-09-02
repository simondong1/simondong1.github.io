# Simon Dong blog style

## Reader-first structure

- Make the mechanism click; do not sound encyclopedic.
- Start from the reader's current mental model and resolve one confusion at a
  time.
- Use the smallest useful concrete example before the general formula.
- Introduce symbols, dimensions, and terminology only when they enter the
  walkthrough. Expand every hardware or kernel acronym on first use. Do not
  assume the reader already knows MMA, TMEM, TMA, SM, warpgroup, CTA,
  occupancy, or similar jargon.
- When a post depends on several GPU or kernel names at once, add a compact
  terms card after the lede. Define only the terms the next section needs;
  introduce the rest when they enter the walkthrough.
- Convert learning questions into a smooth tutorial, not a Q&A transcript.
- End with a compact mental model, primary references, and citation text.

Preferred order:

1. Where the mechanism is used.
2. Why it exists.
3. Minimal baseline or concrete example.
4. General rule or formula.
5. Real model dimensions.
6. Runtime or implementation details that change the mental model.
7. Limitations, variants, and trade-offs.

Do not reteach a mechanism that already has a dedicated post. State its role,
contrast it with the current mechanism, and link the existing article.

## Writing

- Use concise, serious language and short paragraphs.
- Lead with the outcome, then show the evidence.
- Define conventional math symbols immediately. Avoid invented shorthand and
  dense inline algebra.
- A first-time reader must finish the lede without already knowing the
  hardware names. Write “matrix multiply-accumulate (MMA)” before using MMA;
  write “streaming multiprocessor (SM)” before using SM. After the expansion,
  the short form is fine.
- Every blog update requires a first-time-reader proofread. Identify stumbles
  (undefined terms, names used before they are taught, packed sentences)
  and fix them before publishing. Name tiles in prose before the first
  equation that uses them.
- Distinguish mechanism, checkpoint parameterization, model architecture, and
  runtime algorithm.
- Distinguish theoretical optimized layouts from readable reference
  implementations.
- Avoid hype and unsupported model claims.

## Visuals

Prefer, in order:

1. A simple HTML/CSS flow or comparison.
2. A focused chart or small table.
3. A typeset MathJax formula.
4. A short source snippet only when code clarifies an implementation boundary.

Use color consistently. Keep one visual focused on one question. Do not use
large pseudocode blocks, giant profiler tables, or screenshots whose details
the prose does not need.

For performance sections:

- Show wall-clock measurements first.
- Use large, vertically stacked charts; use dots when connecting lines imply a
  trend the data does not establish.
- Explain the cause after the result with only counters that establish it.
- Separate algorithmic gains from hardware- or kernel-specific gains.
- Put exceptions after the main case and explain the smallest metric that
  accounts for the reversal.
- State hardware, dtype, batch, model shape, timing method, and whether the
  result is a kernel microbenchmark or end-to-end.

## Technical checks

- Verify model architecture claims from configs or implementation source.
- Label training, prefill, and decode precisely.
- Include fixed overhead and crossover points when they affect the conclusion.
- Never infer causality from a utilization percentage alone; use total work,
  time, traffic, or launch shape as supporting evidence.
- Do not fabricate timing, traffic, or site analytics.

## Site conventions

Every post needs:

- title, description, canonical URL, author, robots;
- Open Graph and Twitter metadata;
- citation metadata and `TechArticle` JSON-LD;
- publication and modification dates;
- sticky table of contents;
- references and “How to cite this post.”

Keep the homepage, RSS feed, sitemap, `llms.txt`, and page metadata consistent.
Visually inspect desktop, mobile, equations, tables, charts, and interactive
controls before publishing.
