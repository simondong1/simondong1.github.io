"""Render the completed study's article section from audited measurements."""
import argparse
import gzip
import html
import json
from pathlib import Path

LABELS={'ppo':'GRPO (PPO clip)','cispo':'CISPO','glm5':'GLM-5 gate','dppo':'DPPO','sapo':'SAPO','gspo':'GSPO'}
ORDER=list(LABELS)
PREFIX='experiments/rl-surrogates-2026-v2'


def pct(value):return f'{100*value:.2f}%'


def figure(dataset,diagnostic=False):
    stem=f'rl-surrogates-2026-{dataset}-'+('diagnostics' if diagnostic else 'accuracy')
    caption=('Share of responses reaching the cap and prompt groups with identical rewards. Every collection is shown.' if diagnostic else
             'Training: sampled accuracy at collection time, with faint raw values and a trailing five-collection mean. Held out: greedy accuracy on the same 512 questions. One seed; no confidence bands. Accuracy axes use different ranges.')
    return f'''<figure><picture><source media="(max-width: 640px)" srcset="assets/{stem}-mobile.svg"/><img src="assets/{stem}.svg" width="828" height="324" alt="{html.escape(dataset.upper())} {'truncation and reward variance' if diagnostic else 'training and held-out accuracy'} for six RL surrogates"/></picture><figcaption>{caption}</figcaption></figure>'''


def render(summary,results,gates,cases=None):
    assert summary['completed_jobs']==12 and summary['state']=='complete'
    datasets=summary['datasets'];endpoints={task:{e['algorithm']:e for e in v['endpoints']} for task,v in datasets.items()}
    rows=results['runs'];max_memory=max(r['endpoint']['peak_gpu_memory_gib'] for r in rows)
    p=['''<section id="experiments">
<h2>Twelve full-parameter training runs</h2>
<p>We trained <a href="https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236235762e1c871ad0ccb60c8ee5ba337b9a">Qwen3.5-9B</a> in Miles on two B200s: <strong>six surrogates × two math datasets × one seed</strong>. Within each dataset, only the surrogate and its declared thresholds change.</p>
<div class="table-scroll"><table class="ptable settings-table"><tbody>
<tr><th scope="row">Compared</th><td>PPO clipping · CISPO · GLM-5 gating · DPPO · SAPO · GSPO</td></tr>
<tr><th scope="row">Held fixed</th><td>Starting model, prompt order, four responses per prompt, GRPO advantages, sequence mean, optimizer, and update budget</td></tr>
<tr><th scope="row">Full parameters</th><td>All 8.95B text-policy parameters; no adapters. Fresh model and optimizer for every arm.</td></tr>
''']
    p.append(f'<tr><th scope="row">Measured runtime</th><td>{summary["runtime_hours"]:.2f} hours for all twelve jobs; peak sampled memory {max_memory:.1f} GiB on one GPU</td></tr></tbody></table></div>')
    p.append('''<p class="note"><strong>GRPO with PPO clipping is the baseline; no critic.</strong> Actor–critic PPO changes the advantage estimator. DAPO’s similar clip-higher rule is a diagnostic; its full recipe also changes sampling. Thresholds are fixed, not equally tuned.</p>

<h3>GSM8K: little headroom for this checkpoint</h3>
<p><a href="https://huggingface.co/datasets/openai/gsm8k">GSM8K</a> contains grade-school arithmetic word problems. We use 7,409 training questions and a fixed 512-question subset of the official test set. Each arm sees 1,024 prompt presentations: <strong>256 updates, with a 1,536-token response cap</strong>.</p>''')
    p.append(figure('gsm8k'))
    gsm=endpoints['gsm8k'];gbase=datasets['gsm8k']['initial_accuracy_range']
    gfinal=[e['final_accuracy'] for e in gsm.values()]
    gzero=[e['mean_zero_variance_group_fraction'] for e in gsm.values()]
    p.append(f'''<ul class="findings">
<li><strong>The starting policy already scores {pct(gbase[0])}–{pct(gbase[1])}.</strong> Final scores span {pct(min(gfinal))}–{pct(max(gfinal))}.</li>
<li><strong>Most groups provide no relative reward signal:</strong> {pct(min(gzero))}–{pct(max(gzero))} have identical rewards, averaged over each run. This is a weak setting for choosing between surrogates.</li>
</ul>

<h3>DAPO-Math-17K: more room to learn</h3>
<p><a href="https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k">DAPO-Math-17K</a> contains competition math with numeric answers. We deduplicate it and hold out questions before training: 16,088 training questions, with 512 fixed evaluation questions. Each arm sees 1,280 prompt presentations: <strong>320 updates, with a 4,096-token response cap</strong>.</p>''')
    p.append(figure('dapo'))
    dapo=endpoints['dapo'];leader=max(ORDER,key=lambda name:dapo[name]['final_accuracy']);best=dapo[leader]
    p.append(f'''<ul class="findings">
<li><strong>{LABELS[leader]} has the highest final score in this seed:</strong> {pct(best['final_accuracy'])}, versus {pct(dapo['ppo']['final_accuracy'])} for PPO clipping.</li>
<li><strong>The response cap is part of the task.</strong> {LABELS[leader]} reaches it on {pct(best['final_truncation_fraction'])} of final evaluation responses; PPO clipping on {pct(dapo['ppo']['final_truncation_fraction'])}.</li>
</ul>

<div class="table-scroll"><table class="ptable results-table"><thead><tr><th>Surrogate</th><th>GSM8K final</th><th>DAPO-Math final</th></tr></thead><tbody>''')
    for algorithm in ORDER:
        p.append(f'<tr><td>{LABELS[algorithm]}</td><td>{pct(gsm[algorithm]["final_accuracy"])}</td><td>{pct(dapo[algorithm]["final_accuracy"])}</td></tr>')
    p.append('''</tbody></table></div>
<aside class="caveat"><p class="kicker">What these scores mean</p><ul>
<li><strong>One seed is exploratory.</strong> Small differences do not establish an algorithm ranking; repeated greedy baseline evaluations also vary.</li>
<li><strong>We grade the final number, not the proof.</strong> A correct answer can earn reward even if later text reaches the cap.</li>
<li><strong>These are short, partial passes through the data.</strong> They do not establish convergence, performance on unseen domains, or long-horizon agent performance.</li>
</ul></aside>

<details><summary>Learning signal, truncation, and gate activation</summary>
<p>Longer outputs do not create more optimizer steps here. Every arm receives the same number of responses and updates; generated tokens and compute can differ as its policy changes.</p>''')
    for task in ['gsm8k','dapo']:
        p.append(f'<h4>{"GSM8K" if task=="gsm8k" else "DAPO-Math-17K"}</h4>'+figure(task,True))
        vals=list(endpoints[task].values());zeros=[e['zero_gradient_step_fraction'] for e in vals]
        p.append(f'<ul class="findings"><li><strong>{pct(min(zeros))}–{pct(max(zeros))} of minibatches have zero fresh gradient.</strong> Equal group rewards eliminate GRPO advantages; gating can remove further contributions. Adam momentum can still move the weights.</li></ul>')
    p.append('''<p>Apply every rule to the <strong>same PPO-run token probabilities</strong>. The share of nonzero-advantage, sequence-normalized token weight set to zero is:</p>
<div class="table-scroll"><table class="ptable results-table"><thead><tr><th>Rule on PPO inputs</th><th>GSM8K</th><th>DAPO-Math</th></tr></thead><tbody>''')
    for algorithm in ORDER:
        values=[]
        for task in ['gsm8k','dapo']:
            value=next(float(g['conditional_gated_fraction']) for g in gates if g['run']==f'{task}_ppo_s42' and g['counterfactual']==algorithm)
            values.append(pct(value))
        p.append(f'<tr><td>{LABELS[algorithm]}</td><td>{values[0]}</td><td>{values[1]}</td></tr>')
    p.append(f'''</tbody></table></div>
<p>These are counting weights, not fractions of gradient norm. CISPO caps weights without zeroing them; SAPO usually changes their size smoothly. This diagnostic cannot predict an entire counterfactual training run. The <a href="{PREFIX}/gate-audit.csv">full audit</a> includes every trained policy and DAPO clipping.</p>
</details>

<details><summary>Evaluation variability and conditional uncertainty</summary>''')
    for task in ['gsm8k','dapo']:
        row=datasets[task];lo,hi=row['initial_accuracy_range']
        p.append(f'<p><strong>{"GSM8K" if task=="gsm8k" else "DAPO-Math"}:</strong> initial evaluations of the same checkpoint span {pct(lo)}–{pct(hi)}. Mean pairwise disagreement on individual task rewards is {pct(row["mean_initial_pairwise_reward_disagreement"])}. The task IDs and decoding settings match; greedy inference is not bitwise repeatable.</p>')
    p.append(f'''<p>The <a href="{PREFIX}/summary.json">paired task-bootstrap intervals</a> condition on the realized models and decoded outputs. They exclude training-seed and inference-runtime uncertainty. Disjoint study splits also do not rule out pretraining exposure.</p></details>''')
    if cases:
        p.append('''<details><summary>Three actual responses: reaching an answer within the cap</summary>
<p>We inspected the first three zero-to-one reward transitions in GLM-5’s saved DAPO evaluation order. All three initial responses hit 4,096 tokens without a parsable final answer. The trained responses finished with independently checked answers:</p>
<div class="table-scroll"><table class="ptable results-table"><thead><tr><th>Problem</th><th>Trained tokens</th><th>Answer</th></tr></thead><tbody>''')
        names=['Largest three-digit prime factor of C(2000,1000)','Sixth score in seven tests with integer running averages','Smallest GIRLS in MATH + WITH = GIRLS']
        for name,case in zip(names,cases['cases']):
            p.append(f'<tr><td>{name}</td><td>{case["final_length"]:,}</td><td>{case["independently_verified_answer"]}</td></tr>')
        p.append(f'''</tbody></table></div><p>The first answer follows from prime-factor exponents; the other two were checked by exhaustive enumeration. These <a href="{PREFIX}/grader-inspection-cases.json">saved traces</a> illustrate improvement within a response budget, not the frequency of any behavior across the dataset.</p></details>''')
    p.append(f'''

<details id="experiment-recipe"><summary>Frozen recipe, data construction, and correctness checks</summary>
<div class="table-scroll"><table class="ptable settings-table"><tbody>
<tr><th scope="row">Engine</th><td>Miles, Fully Sharded Data Parallel (FSDP2) training and two SGLang engines on the same two B200s</td></tr>
<tr><th scope="row">Collection</th><td>64 prompts × 4 responses; 16 minibatches of 16 responses, consumed once</td></tr>
<tr><th scope="row">Precision</th><td>Bfloat16 computation; 32-bit master weights, gradient reduction, and Adam states</td></tr>
<tr><th scope="row">Optimizer</th><td>AdamW, learning rate 10<sup>−6</sup>, betas 0.9/0.95, no weight decay, gradient-norm limit 1</td></tr>
<tr><th scope="row">Objective</th><td>GRPO with sample-standard-deviation normalization; sequence mean; no reference-policy or entropy penalty</td></tr>
<tr><th scope="row">Decoding</th><td>Training temperature 1, top-p 1; greedy evaluation every 64 updates; thinking template disabled</td></tr>
<tr><th scope="row">Throughput</th><td>8,192-token packing per GPU, gradient checkpointing, up to 128 inference requests per engine</td></tr>
<tr><th scope="row">Surrogate settings</th><td>PPO 0.2/0.2; CISPO and GLM-5 0.5–5; DPPO 0.15; SAPO temperatures 1/1.05; GSPO 0.0003/0.0004</td></tr>
</tbody></table></div>
<p>The rollout anchor remains fixed through all 16 minibatch updates, then weights synchronize. Later minibatches therefore see a changed trainer even in this synchronous recipe.</p>
<h4>Data and grading</h4>
<p>GSM8K uses its official train/test split, with 64 training questions reserved for pilots. DAPO’s public file repeats 17,188 unique questions across 1,791,700 rows. We remove 12 questions with conflicting labels, then reserve 64 for pilots and 1,024 for evaluation; the scheduled evaluations use 512 of those. Splits and hashes are pinned in the <a href="{PREFIX}/data-manifest.json">data manifest</a>.</p>
<p>The reward accepts a final numeric <code>Answer:</code> line or boxed number, including equivalent decimals and thousands separators. A pilot exposed retained end-of-sequence markers breaking extraction; we corrected the parser before all main runs. Every recorded main-run reward replays exactly with the frozen grader.</p>
<h4>Model and numerical checks</h4>
<p>Every final checkpoint contains all 8,953,803,264 text-policy parameters and changed layer-normalization weights in all 32 decoder layers. Unused vision and multi-token-prediction components are outside the trained text policy. Optimizer states were omitted from saved checkpoints to reduce I/O.</p>''')
    mismatches=[r['measurement_audit']['response_mean_abs_logprob_difference'] for r in rows]
    p999=[r['measurement_audit']['token_quantiles']['p999'] for r in rows]
    largest=max(r['measurement_audit']['token_quantiles']['max'] for r in rows)
    above5=sum(r['measurement_audit']['tokens_above']['5'] for r in rows)
    p.append(f'<p>Before each collection’s updates, run-mean absolute trainer/inference log-probability differences are {min(mismatches):.4f}–{max(mismatches):.4f} natural-log units, averaging tokens within each response first. Token-level 99.9th percentiles are {min(p999):.3f}–{max(p999):.3f}; the largest difference is {largest:.3f}, with {above5} tokens above 5. Small averages do not imply identical engines. Optimization drift is logged separately.</p>')
    p.append('''<p>Analytic gradient checks cover the surrogates, and the custom PPO loss matches native Miles on randomized cases. A numerical guard clamps log ratios to [−20,20]; per-update activation is included in the records. Runtime includes setup, compilation, generation, evaluation, and checkpointing, so it is not a loss-kernel speed comparison.</p>
</details>
<div class="closing"><p><strong>Choose the gradient rule, then measure the learning signal.</strong> Reward variance, response budget, and evaluation repeatability determine what this comparison can reveal.</p>''')
    p.append(f'<div class="artifact-links"><a href="https://github.com/simondong1/simondong1.github.io/tree/main/{PREFIX}">Code &amp; reproduction ↗</a><a href="{PREFIX}/endpoints.csv">Final scores ↓</a><a href="{PREFIX}/results.json.gz">Measured records ↓</a></div></div></section>')
    return '\n'.join(p)


def main():
    import csv
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    summary=json.loads((args.data/'summary.json').read_text())
    with gzip.open(args.data/'results.json.gz','rt') as stream:results=json.load(stream)
    with (args.data/'gate-audit.csv').open() as stream:gates=list(csv.DictReader(stream))
    cases_path=args.data/'grader-inspection-cases.json'
    cases=json.loads(cases_path.read_text()) if cases_path.exists() else None
    args.out.write_text(render(summary,results,gates,cases))
    print(args.out)


if __name__=='__main__':main()
