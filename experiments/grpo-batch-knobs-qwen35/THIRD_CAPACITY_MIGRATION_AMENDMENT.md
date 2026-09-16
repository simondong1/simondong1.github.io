# Expand generation after the second durable recovery checkpoint

An additional four-GPU Tier-2 B200 worker was available and prepared while
resume6b trained. Keep the eight training GPUs on their current node and add
four TP1 generation engines, reaching eight engines across two four-GPU nodes.
The resulting allocation is sixteen GPUs across three nodes. All roles remain
synchronous and use separate GPUs.

The previous sixteen eight-engine cycles averaged124.56seconds for generation
and8.04million output tokens per cycle; the first current four-engine cycle
took192.15seconds for8.34million tokens. These are operational estimates with
different policies and samples, not a controlled speed comparison. The
separate inference pilot also favored eight engines. Roughly thirty-four
rollouts remain, so the expected generation saving exceeds the extra startup.

Stop only after resume6b checkpoint93 is archived, retaining96,256responses
and376updates. The first checkpoint92 was restored and audited successfully;
the second cycle additionally provides a warm execution observation before
the move. Restore the same weights, Adam, scheduler, trainer RNG, data position
and sample/token counters. Keep P128/G8/B256/S4, LR1e-6, response cap16384,
packing8192 and fixed131072-response endpoint. Recovery saves remain every1024
responses; validation every8192; selection restricted to the initial model
and32768-response multiples. No other learning setting changes.

Explicit training-node priority keeps all eight trainers and checkpoint shards
on the same host. The existing per-host serving allocator handles the two
four-GPU generation hosts. Verify actual bundle order before accepting work.
The prior raw records and complete job time remain in execution accounting.
