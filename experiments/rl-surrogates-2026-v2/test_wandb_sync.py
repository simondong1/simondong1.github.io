"""Logging integrity checks; no credentials, network or GPUs required."""
import json
from pathlib import Path
import tempfile
import unittest

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from wandb_sync import Collector, recover_journal


CONFIG = {'prompts_per_rollout': 64, 'group_size': 4, 'global_batch_size': 16}


class SyncIntegrity(unittest.TestCase):
    def test_delayed_writers_preserve_metrics_and_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = EventFileWriter(str(root / 'tensorboard'))
            def emit(tag, value, step):
                writer.add_event(Event(wall_time=100 + step, step=step,
                                       summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)])))
                writer.flush()
            try:
                emit('train/loss', .25, 0)
                collector = Collector(root, CONFIG, 100, {})
                first = collector.pending()
                self.assertEqual(first[0]['data']['axis/optimizer_updates'], 1)
                for row in first:
                    collector.commit(row)
                self.assertEqual(collector.pending(), [])
                # A delayed metric at the same step must still be imported.
                emit('train/grad_norm', .5, 0)
                emit('rollout/entropy', .75, 0)
                late = collector.pending()
                self.assertEqual(len(late), 2)
                self.assertFalse(any('train/loss' in row['data'] for row in late))
                self.assertTrue(any(row['data'].get('axis/native_rollout_step') == 0 for row in late))
            finally:
                writer.close()

    def test_collection_is_not_counted_as_completed_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'measurements').mkdir()
            path = root / 'measurements/rollouts.jsonl'
            item = {'time': 101, 'rollout_id': 2, 'optimizer_updates': 48,
                    'generated_tokens': 100, 'rollout_seconds': 10, 'reward_mean': .5}
            encoded = json.dumps(item)
            path.write_text(encoded[:20])
            collector = Collector(root, CONFIG, 100, {})
            self.assertEqual(collector.pending(), [])
            with path.open('a') as stream:
                stream.write(encoded[20:] + '\n')
            rows = collector.pending()
            self.assertEqual(rows[0]['data']['axis/sampling_policy_updates'], 32)
            self.assertNotIn('sampling/optimizer_updates', rows[0]['data'])
            collector.commit(rows[0])
            self.assertEqual(collector.pending(), [])

    def test_journal_recovers_unsent_records_and_partial_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'journal.jsonl'
            records = [{'marks': {'file_a': 12}, 'data': {'x': 1}},
                       {'marks': {'file_a': 18, 'file_b': 5}, 'data': {'x': 2}}]
            complete = ''.join(json.dumps(row) + '\n' for row in records)
            path.write_text(complete + '{"marks":')
            cursor, recovered = recover_journal(path)
            self.assertEqual(recovered, records)
            self.assertEqual(cursor, {'file_a': 18, 'file_b': 5})
            self.assertEqual(path.read_text(), complete)


if __name__ == '__main__':
    unittest.main()
