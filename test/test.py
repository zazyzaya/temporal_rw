import temporal_walks
import torch


print("Loading graph")
g = torch.load('/home/isaiah/code/GraphGPT/data/lanl_tgraph_tr.pt', weights_only=False)
print("Done")
print()

for device in [0, 'cpu']:
    print('#' * 10, device, '#' * 10)
    g.idxptr = g.idxptr.to(device)
    g.col = g.col.to(device)
    g.ts = g.ts.to(device)
    start = torch.randint(0, 7000, (4,)).to(device)

    rw, ei = temporal_walks.temporal_rw(
        g.idxptr, g.col, g.ts, start, 4,
        return_edge_indices=True,
    )
    ts = g.ts[ei]
    ts[ei == -1] = -1
    print('Normal')
    print(rw)
    print(ts)
    print()

    rw, ei = temporal_walks.temporal_rw(
        g.idxptr, g.col, g.ts, start, 4,
        return_edge_indices=True, reverse=True
    )
    ts = g.ts[ei]
    ts[ei == -1] = -1
    print('Reverse')
    print(rw)
    print(ts)
    print()

    rw, ei = temporal_walks.temporal_rw(
        g.idxptr, g.col, g.ts, start, 4,
        return_edge_indices=True,
        min_ts=0,
        max_ts=86000
    )
    ts = g.ts[ei]
    ts[ei == -1] = -1
    print('Bounded')
    print(rw)
    print(ts)

    rw, ei = temporal_walks.temporal_rw(
        g.idxptr, g.col, g.ts, start, 4,
        return_edge_indices=True,
        min_ts=0,
        max_ts=86000,
        reverse=True
    )
    ts = g.ts[ei]
    ts[ei == -1] = -1
    print('Bounded (rev)')
    print(rw)
    print(ts)


    st = torch.randint_like(start, 58*24*60*60 // 2)
    en = torch.randint_like(start, 58*24*60*60)
    rw, ei = temporal_walks.temporal_rw(
        g.idxptr, g.col, g.ts, start, 4,
        return_edge_indices=True,
        min_ts=st,
        max_ts=en
    )
    ts = g.ts[ei]
    ts[ei == -1] = -1
    print('continuous')
    print(rw)
    print(ts)

    rw, ei = temporal_walks.temporal_rw(
        g.idxptr, g.col, g.ts, start, 4,
        return_edge_indices=True,
        min_ts=st,
        max_ts=en,
        reverse=True
    )
    ts = g.ts[ei]
    ts[ei == -1] = -1
    print('continuous (rev)')
    print(rw)
    print(ts)

    print('#' * 25)