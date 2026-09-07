"""
II (item-item) co-occurrence graph built from user interaction sequences.

For each source node, interactions are sorted by time. Consecutive destination
items in the sorted sequence form directed edges (earlier → later). The global
graph over all destination items captures collaborative transition patterns.

The graph is directed and acyclic with respect to interaction time, so there is
no future information leakage when propagating along edge direction.
"""

import numpy as np
import scipy.sparse as sp
from collections import defaultdict


def build_ii_graph(src_np, dst_np, t_np, n_dst=None, return_user_edges=False):
    """Build a directed II graph from interaction triples.

    Args:
        src_np: (N,) int32, source node ids.
        dst_np: (N,) int32, destination node ids.
        t_np: (N,) int32, interaction timestamps.
        n_dst: int or None, number of dst nodes. If None, inferred from dst_np.
        return_user_edges: bool, if True also return per-occurrence edges with
            the user (src) that created each edge, for dynamic attention.

    Returns:
        dict with keys:
            head_list: (E,) int64, source node of each edge.
            tail_list: (E,) int64, target node of each edge.
            edge_weights: (E,) float32, normalized edge weight.
            in_degree: (n_dst,) float32, sum of incoming edge weights.
            adj_csr: scipy.sparse.csr_matrix (n_dst × n_dst), mean-normalized.
            (if return_user_edges):
                user_head: (E_raw,) int64, per-occurrence head nodes.
                user_tail: (E_raw,) int64, per-occurrence tail nodes.
                user_list: (E_raw,) int64, per-occurrence user (src) nodes.
                user_in_deg: (n_dst,) float32, per-occurrence in-degree.
    """
    if n_dst is None:
        n_dst = int(dst_np.max()) + 1

    # --- group interactions by src, then sort each group by time ---
    src_order = np.argsort(src_np, kind='stable')
    src_sorted = src_np[src_order]
    dst_sorted = dst_np[src_order]
    t_sorted = t_np[src_order]

    # Find boundaries of each src group
    unique_src, src_starts, src_counts = np.unique(
        src_sorted, return_index=True, return_counts=True)

    # Build edge pairs with co-occurrence counts
    edge_counts = defaultdict(int)
    user_heads, user_tails, user_users = [], [], []  # per-occurrence

    for i in range(len(unique_src)):
        start = src_starts[i]
        count = src_counts[i]
        if count < 2:
            continue
        user_id = unique_src[i]
        # Get this src's dst sequence in global time order
        seq_dst = dst_sorted[start:start + count]
        seq_t = t_sorted[start:start + count]

        # Sort by time within this src
        time_order = np.argsort(seq_t, kind='stable')
        seq_dst = seq_dst[time_order]

        # Add directed edges for consecutive pairs
        for h, t in zip(seq_dst[:-1], seq_dst[1:]):
            if h == t:  # skip self-loops
                continue
            edge_counts[(h, t)] += 1
            if return_user_edges:
                user_heads.append(h)
                user_tails.append(t)
                user_users.append(user_id)

    if len(edge_counts) == 0:
        raise RuntimeError('No II edges built — check that at least one src '
                           'has ≥ 2 interactions at different timestamps.')

    # --- build sparse adjacency matrix ---
    pairs, counts = zip(*edge_counts.items())
    heads, tails = zip(*pairs)
    heads = np.array(heads, dtype=np.int64)
    tails = np.array(tails, dtype=np.int64)
    counts = np.array(counts, dtype=np.float32)

    adj_raw = sp.csr_matrix((counts, (tails, heads)), shape=(n_dst, n_dst))

    # --- in-degree (mean) normalization:  D_in^{-1} * A ---
    d_in = np.asarray(adj_raw.sum(axis=1)).flatten()
    d_in_clamped = np.where(d_in > 0, d_in, 1.0)
    rec_d_in = 1.0 / d_in_clamped
    rec_d_in[d_in == 0] = 0.0
    D_inv = sp.diags(rec_d_in)
    adj_norm = (D_inv @ adj_raw).tocsr()
    adj_norm.eliminate_zeros()

    # Extract COO information for Jittor propagation
    adj_coo = adj_norm.tocoo()
    edge_weights = adj_coo.data.astype(np.float32)
    tail_list = adj_coo.row.astype(np.int64)  # COO rows = target nodes
    head_list = adj_coo.col.astype(np.int64)  # COO cols = source nodes

    result = {
        'head_list': head_list,
        'tail_list': tail_list,
        'edge_weights': edge_weights,
        'in_degree': d_in.astype(np.float32),
        'adj_csr': adj_norm,
        'n_nodes': n_dst,
        'num_edges': len(head_list),
    }

    if return_user_edges:
        user_heads = np.array(user_heads, dtype=np.int64)
        user_tails = np.array(user_tails, dtype=np.int64)
        user_users = np.array(user_users, dtype=np.int64)
        # Per-occurrence in-degree (for attention normalization)
        user_d_in = np.bincount(user_tails, minlength=n_dst).astype(np.float32)
        result['user_head'] = user_heads
        result['user_tail'] = user_tails
        result['user_list'] = user_users
        result['user_num_edges'] = len(user_heads)
        result['user_in_deg'] = user_d_in

    return result


def print_graph_stats(graph):
    """Print summary statistics of an II graph."""
    n_nodes = graph['n_nodes']
    num_edges = graph['num_edges']
    d_in = graph['in_degree']
    nonzero = (d_in > 0).sum()
    print(f'II Graph: {n_nodes} nodes, {num_edges} unique edges')
    if 'user_num_edges' in graph:
        print(f'  Per-occurrence edges: {graph["user_num_edges"]}')
    print(f'  Density: {num_edges / (n_nodes * n_nodes):.6f}')
    print(f'  Nodes with in-degree > 0: {nonzero} / {n_nodes} '
          f'({100 * nonzero / n_nodes:.1f}%)')
    print(f'  Mean in-degree: {d_in.sum() / nonzero:.1f}')
    print(f'  Median in-degree: {np.median(d_in[d_in > 0]):.1f}')
    print(f'  Max in-degree: {d_in.max():.0f}')

class IncrementalUIGraph:
    """UI graph built incrementally as training progresses.

    Edges are accumulated batch-by-batch. Call add_edges() after each
    training batch, and build_ui_data() to get LightGCN-ready tensors.
    This prevents future information leakage.
    """

    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.adj = sp.lil_matrix((n_nodes, n_nodes), dtype=np.float32)
        self.total_deg = np.zeros(n_nodes, dtype=np.float32)

    def reset(self):
        """Clear all accumulated edges."""
        self.adj = sp.lil_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        self.total_deg = np.zeros(self.n_nodes, dtype=np.float32)

    def add_edges(self, src_np, dst_np):
        """Add bidirectional edges for a batch of interactions."""
        src = np.asarray(src_np, dtype=np.int64).reshape(-1)
        dst = np.asarray(dst_np, dtype=np.int64).reshape(-1)
        for u, i in zip(src, dst):
            self.adj[u, i] += 1.0
            self.adj[i, u] += 1.0
        self.total_deg[src] += 1.0
        self.total_deg[dst] += 1.0

    def build_ui_data(self, dropout=0.0):
        adj_csr = self.adj.tocsr()

        # ---- Random edge dropout (FREEDOM-style) ----
        if dropout > 0:
            coo = adj_csr.tocoo()
            n_edges = len(coo.data)
            weights = np.abs(coo.data).astype(np.float64)
            weights /= weights.sum()
            keep_len = int(n_edges * (1.0 - dropout))
            keep_idx = np.random.choice(n_edges, size=max(keep_len, 1),
                                        replace=False, p=weights)
            adj_csr = sp.csr_matrix(
                (coo.data[keep_idx], (coo.row[keep_idx], coo.col[keep_idx])),
                shape=adj_csr.shape)

        d_inv_sqrt = np.zeros_like(self.total_deg)
        mask = self.total_deg > 0
        d_inv_sqrt[mask] = 1.0 / np.sqrt(self.total_deg[mask])
        D_inv = sp.diags(d_inv_sqrt)
        adj_norm = D_inv @ adj_csr @ D_inv
        adj_norm.eliminate_zeros()

        coo = adj_norm.tocoo()
        return {
            'head_list': coo.col.astype(np.int64),
            'tail_list': coo.row.astype(np.int64),
            'edge_weights': coo.data.astype(np.float32),
            'degree': self.total_deg.astype(np.float32),
            'n_nodes': self.n_nodes,
            'num_edges': len(coo.data),
        }

