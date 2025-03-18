from typing import Optional, Tuple, Union

import torch
from torch import Tensor


def temporal_rw(
    rowptr: Tensor,
    col: Tensor,
    ts: Tensor,
    start: Tensor,
    walk_length: int,
    return_edge_indices: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Samples random walks of length :obj:`walk_length` from all node indices
    in :obj:`start` in the graph given by :obj:`(row, col)` s.t. each edge is
    older than the previous one sampled.

    WARNING this method does not baby you. It's just a hook into the underlying
    torch cpp/cuda implementation. It requires that the input is in CSR
    format already, and timestamps are sorted per node, e.g.
        ts[rowptr[i]]:ts[rowptr[i+1]] are strictly ascending in order
    Also (for now) timecodes have to be ints

    Args:
        row (LongTensor): Source nodes.
        col (LongTensor): Target nodes.
        ts (LongTensor): Timestamp of edges.
        start (LongTensor): Nodes from where random walks start.
        walk_length (int): The walk length.
        p (float, optional): Likelihood of immediately revisiting a node in the
            walk. (default: :obj:`1`)
        q (float, optional): Control parameter to interpolate between
            breadth-first strategy and depth-first strategy (default: :obj:`1`)
        coalesced (bool, optional): If set to :obj:`True`, will coalesce/sort
            the graph given by :obj:`(row, col)` according to :obj:`row`.
            (default: :obj:`True`)
        num_nodes (int, optional): The number of nodes. (default: :obj:`None`)
        return_edge_indices (bool, optional): Whether to additionally return
            the indices of edges traversed during the random walk.
            (default: :obj:`False`)

    :rtype: :class:`LongTensor`
    """
    node_seq, edge_seq = torch.ops.torch_cluster.temporal_random_walk(
        rowptr, col, ts, start, walk_length)

    if return_edge_indices:
        return node_seq, edge_seq

    return node_seq
