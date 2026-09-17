"""Let the pinned port allocator use the rollout nodes' actual GPU count.

Training and serving may have different node sizes. Physical GPU IDs already
come from Ray placement; this changes only serving address/port allocation.
"""
import copy
import os


def install():
    if os.environ.get("STUDY_HETEROGENEOUS_ROLLOUT") != "1":
        return
    import ray
    from miles.ray.rollout import server_group

    native = server_group.allocate_rollout_engine_addr_and_ports_normal
    if getattr(native, "_study_topology", False):
        return
    def allocate(**kwargs):
        engines = kwargs["rollout_engines"]
        actual = ray.get([engine._get_current_node_ip_and_free_port.remote() for _, engine in engines])
        groups = {}
        for item, address in zip(engines, actual):
            groups.setdefault(address[0], []).append(item)
        result = {}
        cursors = server_group.PortCursors.empty()
        for node_index, group in enumerate(groups.values()):
            args = copy.copy(kwargs["args"])
            assert (kwargs.get("num_gpus_per_engine") or args.rollout_num_gpus_per_engine) == 1
            ranks = [rank for rank, _ in group]
            assert ranks == list(range(ranks[0], ranks[0] + len(ranks)))
            args.num_gpus_per_node = len(group)
            values, node_cursors = native(**{**kwargs, "args": args,
                                            "rollout_engines": group, "rank_offset": ranks[0]})
            result.update(values)
            cursors._values[node_index] = node_cursors.next_base_port()
        assert all(result[rank]["host"] == address[0] for (rank, _), address in zip(engines, actual))
        return result, cursors

    allocate._study_topology = True
    server_group.allocate_rollout_engine_addr_and_ports_normal = allocate
