"""Summarize held-out comparisons without pooling repeated prompts as new data."""
import argparse
import json
from pathlib import Path
import numpy as np
from make_figures import load, paired_prompt_interval


def crossing(run, threshold):
    previous = None
    for point in run['curves']:
        if point['accuracy'] >= threshold:
            return {'observed': True, 'sample_bracket': [previous['samples'] if previous else 0, point['samples']],
                    'phase_seconds_bracket': [previous['active_phase_seconds'] if previous else 0, point['active_phase_seconds']],
                    'job_seconds_bracket': [previous['job_elapsed_seconds'] if previous else 0, point['job_elapsed_seconds']]}
        previous = point
    return {'observed': False, 'last_samples': previous['samples'], 'last_phase_seconds': previous['active_phase_seconds']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = load(args.results)
    plan = json.loads(args.plan.read_text())
    lookup = {r['name']: r for r in runs}
    output = {'uncertainty': 'Paired task-ID bootstrap conditional on the three trained seeds; seed spread reported separately.',
              'time_axis': 'Sum of recorded generation, trainer, and weight-sync durations. Complete job time is separate.',
              'initial_model_test': {task: lookup[f'{task}-initial-test']['summary'] for task in ['gsm8k', 'countdown']},
              'candidate_comparisons': {}, 'tasks': {}}
    for task, arms in plan['selection']['tasks'].items():
        task_results = {}
        test_predictions = {}
        for arm in arms:
            selected = [r for r in runs if r['config']['phase'] == 'confirm' and r['config']['task'] == task and r['config']['arm'] == arm]
            selected.sort(key=lambda r: r['config']['seed'])
            assert [r['config']['seed'] for r in selected] == [1, 2, 3]
            test_predictions[arm] = {}
            records = []
            for run in selected:
                test = lookup[run['name']+'-test']
                assert test['config']['training_run'] == run['name']
                seed = run['config']['seed']
                test_predictions[arm][seed] = {r['id']: r['reward'] for r in test['examples']}
                assert len(test_predictions[arm][seed]) == test['summary']['n']
                records.append({'seed': seed, **run['summary'], 'test_accuracy': test['summary']['accuracy'],
                                'test_mean_response_length': test['summary']['mean_response_length'],
                                'test_truncation_fraction': test['summary']['truncation_fraction'],
                                'test_evaluation_seconds': test['summary']['job_seconds'],
                                'targets': {str(t): crossing(run,t) for t in plan['targets'][task]}})
            keys = ['final_accuracy', 'test_accuracy', 'sample_auc', 'active_phase_seconds', 'job_seconds',
                    'mixed_group_fraction', 'mean_response_length', 'output_tokens',
                    'generation_seconds', 'training_seconds', 'weight_sync_seconds',
                    'prompt_exposures', 'distinct_training_prompts', 'test_mean_response_length', 'test_truncation_fraction']
            task_results[arm] = {'shape': selected[0]['config']['shape'], 'lr': selected[0]['config']['lr'],
                                 'per_seed': records, 'mean': {k: float(np.mean([r[k] for r in records])) for k in keys},
                                 'seed_min_max': {k: [min(r[k] for r in records),max(r[k] for r in records)] for k in keys}}
        for arm in arms:
            if arm != 'onpolicy-small':
                task_results[arm]['test_difference_vs_baseline'] = paired_prompt_interval(test_predictions[arm], test_predictions['onpolicy-small'])
        candidates = [arm for arm in arms if arm != 'onpolicy-small']
        assert len(candidates) == 2
        reference, candidate = candidates
        output['candidate_comparisons'][task] = {
            'candidate': candidate, 'reference': reference,
            **paired_prompt_interval(test_predictions[candidate], test_predictions[reference])}
        output['tasks'][task] = task_results
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps({task: {arm: {'mean': r['mean'], 'paired': r.get('test_difference_vs_baseline')} for arm,r in data.items()}
                      for task,data in output['tasks'].items()}, indent=2))


if __name__ == '__main__':
    main()
