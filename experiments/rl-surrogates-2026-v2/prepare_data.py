"""Use pinned public releases; split DAPO by unique question before sampling."""
import hashlib
import json
from pathlib import Path
import random
import re
import pyarrow.parquet as pq

ROOT = Path('/tmp/rl-surrogate-2026-v2')
SOURCES = {
    'gsm8k': {'repo': 'openai/gsm8k', 'revision': '740312add88f781978c0658806c59bc2815b9866'},
    'dapo': {'repo': 'BytedTsinghua-SIA/DAPO-Math-17k', 'revision': '65877096c24ffa7abc4e4fa5edb95cf3413a5674'},
}


def encode(task, question, answer):
    identity = ' '.join(question.split())
    uid = hashlib.sha256(identity.encode()).hexdigest()
    return {'prompt': [{'role': 'user', 'content': question}],
            'label': json.dumps({'task': task, 'answer': answer}),
            'metadata': {'uid': uid, 'dataset': task}}


def prepare():
    out = ROOT/'data'; out.mkdir(exist_ok=True)
    manifest = {'seed': 20260915, 'sources': SOURCES, 'splits': {}}
    groups = {}
    for split in ['train', 'test']:
        rows = pq.read_table(ROOT/'raw'/f'gsm8k_{split}.parquet').to_pylist()
        unique = {}
        for row in rows:
            answer = row['answer'].rsplit('####', 1)[1].strip().replace(',', '')
            assert re.fullmatch(r'[+-]?\d+(?:\.\d+)?', answer), answer
            q = row['question'] + '\nSolve the problem. Give concise reasoning, then write your final numeric answer on its own line as Answer: <number>.'
            item = encode('gsm8k', q, answer); key=item['metadata']['uid']
            assert key not in unique or unique[key]['label']==item['label']
            unique[key] = item
        groups['gsm8k_'+split] = list(unique.values())
    # This HF parquet repeats the 17K question collection many times.
    unique = {}; total = 0; conflicts = {}
    for batch in pq.ParquetFile(ROOT/'raw'/'dapo.parquet').iter_batches(batch_size=8192, columns=['prompt','reward_model']):
        for row in batch.to_pylist():
            total += 1
            answer = str(row['reward_model']['ground_truth']).strip()
            assert re.fullmatch(r'[+-]?\d+(?:\.\d+)?', answer), answer
            item = encode('dapo', row['prompt'][0]['content'], answer);key=item['metadata']['uid']
            if key in unique and unique[key]['label'] != item['label']:
                conflicts.setdefault(key, {'prompt': item['prompt'], 'answers': set()})['answers'].update(
                    [json.loads(unique[key]['label'])['answer'], answer])
            unique[key]=item
    unique_count = len(unique)
    for key in conflicts: unique.pop(key)
    (out/'dapo_conflicts.json').write_text(json.dumps(
        {k:{**v,'answers':sorted(v['answers'])} for k,v in conflicts.items()},indent=2))
    manifest['sources']['dapo'].update(raw_rows=total, unique_questions=unique_count,
        duplicates_removed=total-unique_count, conflicting_questions_removed=len(conflicts))
    dapo = list(unique.values());random.Random(20260915).shuffle(dapo)
    gsm = groups['gsm8k_train'];random.Random(20260915).shuffle(gsm)
    gsmtest = groups['gsm8k_test'];random.Random(20260915).shuffle(gsmtest)
    splits = {'gsm8k_train':gsm[64:], 'gsm8k_pilot':gsm[:64], 'gsm8k_test':gsmtest[:512],
              'gsm8k_test_full':gsmtest, 'dapo_train':dapo[1088:], 'dapo_pilot':dapo[:64],
              'dapo_test':dapo[64:576], 'dapo_test_full':dapo[64:1088]}
    for task in ['gsm8k','dapo']:
        sets=[{x['metadata']['uid'] for x in splits[task+'_'+s]} for s in ['train','pilot','test_full']]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    for name,rows in splits.items():
        path=out/(name+'.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest['splits'][name]={'count':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': prepare()
