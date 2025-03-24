import temporal_walks
import torch


print("Loading graph")
g = torch.load('/home/isaiah/code/GraphGPT/data/lanl_tgraph_tr.pt', weights_only=False)
print("Done")
print()

g.idxptr = g.idxptr.to(0)
g.col = g.col.to(0)
g.ts = g.ts.to(0)
start = torch.tensor([12199, 12766, 11021]).to(0)

rw, ei = temporal_walks.temporal_rw(
    g.idxptr, g.col, g.ts, start, 8,
    return_edge_indices=True,
)
ts = g.ts[ei]
ts[ei == -1] = -1
print('Normal')
print(rw)
print(ts)
print()

rw, ei = temporal_walks.temporal_rw(
    g.idxptr, g.col, g.ts, start, 8,
    return_edge_indices=True, reverse=True
)
ts = g.ts[ei]
ts[ei == -1] = -1
print('Reverse')
print(rw)
print(ts)
print()

rw, ei = temporal_walks.temporal_rw(
    g.idxptr, g.col, g.ts, start, 8,
    return_edge_indices=True,
    min_ts=0,
    max_ts=86400
)
ts = g.ts[ei]
ts[ei == -1] = -1
print('1k - 5k')
print(rw)
print(ts)

rw, ei = temporal_walks.temporal_rw(
    g.idxptr, g.col, g.ts, start, 8,
    return_edge_indices=True,
    min_ts=0,
    max_ts=86400,
    reverse=True
)
ts = g.ts[ei]
ts[ei == -1] = -1
print('5k - 1k (rev)')
print(rw)
print(ts)
