import jittor as jt
from jittor import nn
import math
import copy
import numpy as np
from jittor.nn import Embedding

class TimeProjection(nn.Module):
    """MLP for time interval projection."""

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def execute(self, x):
        return self.dropout(self.norm(self.net(x.float().view(-1, 1))))


class TimeBucketEmbed(nn.Module):
    """Discrete time bucketing via binary-search on custom boundaries."""

    def __init__(self, hidden_size, boundaries):
        super().__init__()
        self.boundaries = jt.array(boundaries.astype(np.float32)).stop_grad()
        self.emb = nn.Embedding(len(boundaries) + 1, hidden_size)

    def execute(self, dt):
        dt_np = dt.float().abs().numpy()
        idx_np = np.searchsorted(self.boundaries.numpy(), dt_np)
        return self.emb(jt.array(idx_np)).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Token Mixer (RankMixer-style)
# ═════════════════════════════════ Token Mixer (MLP-Mixer style) ═══════════════

class TokenMixerLayer(nn.Module):
    """Sequence encoder: channel + token mixing, O(LD) instead of O(L^2 D)."""

    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.channel_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )
        self.token_interact = nn.Sequential(
            nn.Linear(d_model * 2, d_model * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def execute(self, x, key_padding_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.channel_ffn(x)
        x = residual + x
        residual = x
        x = self.norm2(x)
        if key_padding_mask is not None:
            mask = (1 - key_padding_mask.float()).unsqueeze(-1)
            gmean = (x * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp(min_v=1.0)
        else:
            gmean = x.mean(dim=1, keepdim=True)
        x = jt.cat([x, gmean.expand_as(x)], dim=-1)
        x = self.token_interact(x)
        return residual + x


# ═══════════════════════════════════════════════════════════════════════════════

class FeatTokenMixer(nn.Module):
    """Feature token mixing: per-token FFN + residual, then mean-pool across features."""

    def __init__(self, d_model, n_tokens, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)

    def execute(self, x):
        # x: (B, C, N, D)  C=candidates, N=features, D=hidden
        # Per-feature FFN (shared params) + residual
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = nn.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = residual + x
        # Mean pool across features → (B, C, D)
        return x.mean(dim=2)
# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Attention (copied from official CRAFT model_ori.py)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiHeadCrossAttentionbyHand(nn.Module):
    def __init__(self, n_heads, hidden_size, hidden_dropout_prob,
                 attn_dropout_prob, layer_norm_eps):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} not divisible by n_heads {n_heads}")
        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_attention_head_size = math.sqrt(self.attention_head_size)
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return x.view(*new_shape)

    def execute(self, query, attention_mask, key=None):
        if key is None:
            key = query
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(key)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        query_layer = query_layer.permute(0, 2, 1, 3)
        key_layer = key_layer.permute(0, 2, 3, 1)
        value_layer = value_layer.permute(0, 2, 1, 3)
        attention_scores = jt.matmul(query_layer, key_layer)
        attention_scores = attention_scores / self.sqrt_attention_head_size
        attention_scores = attention_scores + attention_mask
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = jt.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = hidden_states + query
        return hidden_states


class FeedForward4CrossAttn(nn.Module):
    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act,
                 layer_norm_eps, output_dim=None):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = nn.relu if hidden_act == 'relu' else nn.gelu
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        if output_dim is not None:
            self.output_dim = output_dim
            self.dense_2 = nn.Linear(inner_size, output_dim)
        else:
            self.output_dim = -1
            self.dense_2 = nn.Linear(inner_size, hidden_size)
            self.dropout = nn.Dropout(hidden_dropout_prob)

    def execute(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        if self.output_dim == -1:
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.LayerNorm(hidden_states)
            hidden_states = hidden_states + input_tensor
        return hidden_states


class CrossAttentionLayer(nn.Module):
    def __init__(self, n_heads, hidden_size, intermediate_size,
                 hidden_dropout_prob, attn_dropout_prob, hidden_act,
                 layer_norm_eps, output_dim=None):
        super().__init__()
        self.multi_head_attention = MultiHeadCrossAttentionbyHand(
            n_heads, hidden_size, hidden_dropout_prob,
            attn_dropout_prob, layer_norm_eps)
        self.feed_forward = FeedForward4CrossAttn(
            hidden_size, intermediate_size, hidden_dropout_prob,
            hidden_act, layer_norm_eps, output_dim=output_dim)

    def execute(self, query, attention_mask, key=None):
        attention_output = self.multi_head_attention(query, attention_mask, key)
        return self.feed_forward(attention_output)


class CrossAttention(nn.Module):
    def __init__(self, n_layers=2, n_heads=2, hidden_size=64, inner_size=256,
                 hidden_dropout_prob=0.1, attn_dropout_prob=0.1, hidden_act="relu",
                 layer_norm_eps=1e-12, output_dim=None):
        super().__init__()
        layer = CrossAttentionLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob,
            attn_dropout_prob, hidden_act, layer_norm_eps, output_dim=output_dim)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def execute(self, query, attention_mask, key=None, output_all_encoded_layers=True):
        for layer_module in self.layer:
            query = layer_module(query, attention_mask, key)
        return query


class TNCNModel(nn.Module):
    """TNCN model for temporal link prediction.

    Combines cross-attention over temporal neighbor sequences with
    NCN (common neighbor) structural features.
    """

    def __init__(
        self,
        n_nodes,
        hidden_size=64,
        n_layers=2,
        n_heads=2,
        dropout=0.1,
        output_cat_repeat_times=False,
        use_dst_neighbor_pooling=False,
        use_seq_self_attn=False,
        seq_encoder_layers=1,
        use_feat_mixer=True,
        use_pop_feat=True,
        use_infonce=False,
        lambda_infonce=0.0,
        infonce_temperature=0.07,
        temperature=1.0,
        lambda_l2=0.0,
        ii_graph=None,
        ii_layers=2,
        ii_use_attn=False,
        ii_attn_tau=1.0,
        ii_layer_weight=False,
        ui_graph=None,
        ui_layers=0,
        time_bucket_boundaries=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_cat_repeat_times = output_cat_repeat_times
        self.use_dst_neighbor_pooling = use_dst_neighbor_pooling
        self.use_seq_self_attn = use_seq_self_attn
        self.seq_encoder_layers = seq_encoder_layers
        self.temperature = temperature
        self.lambda_l2 = lambda_l2
        self.node_embedding = Embedding(int(n_nodes) + 1, hidden_size)

        # II (item-item) co-occurrence graph propagation
        self.ii_layers = ii_layers
        self.ii_use_attn = ii_use_attn
        self.ii_attn_tau = ii_attn_tau
        self.ii_layer_weight = ii_layer_weight
        if ii_graph is not None:
            self.ii_head = jt.array(ii_graph['head_list']).stop_grad()
            self.ii_tail = jt.array(ii_graph['tail_list']).stop_grad()
            self.ii_weight = jt.array(ii_graph['edge_weights']).stop_grad()
            self.ii_in_deg = jt.array(ii_graph['in_degree']).float().stop_grad()
            self.ii_in_deg_clamped = self.ii_in_deg.clamp(min_v=1.0).stop_grad()
            self.ii_n_nodes = ii_graph['n_nodes']
            # Per-occurrence user edges for dynamic attention
            if ii_use_attn and 'user_list' in ii_graph:
                self.ii_user_head = jt.array(ii_graph['user_head']).stop_grad()
                self.ii_user_tail = jt.array(ii_graph['user_tail']).stop_grad()
                self.ii_user_list = jt.array(ii_graph['user_list']).stop_grad()
                self.ii_user_in_deg = jt.array(ii_graph['user_in_deg']).float().stop_grad()
                self.ii_user_in_deg_clamped = self.ii_user_in_deg.clamp(min_v=1.0).stop_grad()
        else:
            self.ii_head = None

        # UI (user-item) interaction graph propagation (LightGCN-style)
        self.ui_layers = ui_layers
        self.ui_n_nodes = int(n_nodes)
        if ui_graph is not None:
            self._set_ui_tensors(ui_graph)
        else:
            self.ui_head = None

        self.use_feat_mixer = use_feat_mixer
        self.use_pop_feat = use_pop_feat
        self.use_infonce = use_infonce
        self.lambda_infonce = lambda_infonce
        self.infonce_temperature = infonce_temperature
        
        self.time_bucket = TimeBucketEmbed(hidden_size,
            boundaries=time_bucket_boundaries if time_bucket_boundaries is not None
                        else np.logspace(0, 8, 64).astype(np.int64))
        if use_pop_feat:
            self.degree_proj = TimeProjection(hidden_size, dropout)
            self.degree_norm = nn.LayerNorm(hidden_size)
        if output_cat_repeat_times:
            self.repeat_times_proj = TimeProjection(hidden_size, dropout)
            self.repeat_times_norm = nn.LayerNorm(hidden_size)

        # Token mixer encoder (MLP-Mixer style, replaces self-attention)
        self.use_token_mixer = use_seq_self_attn
        self.token_mixer_layers = seq_encoder_layers
        if self.use_token_mixer:
            self.token_mixer = nn.ModuleList([
                TokenMixerLayer(hidden_size, dropout=dropout)
                for _ in range(seq_encoder_layers)
            ])

        # Cross-attention (official CRAFT architecture)
        self.cross_attn = CrossAttention(
            n_layers=n_layers, n_heads=n_heads, hidden_size=hidden_size,
            inner_size=hidden_size * 4, hidden_dropout_prob=dropout,
            attn_dropout_prob=dropout, hidden_act="gelu", layer_norm_eps=1e-12,
        )

        # Feature mixing: treat each feature type as a token, apply RankMixer
        num_feat = 1  # attn_out, dst_time_feat, cn_features, src_emb_expand, src_xij
        if output_cat_repeat_times:
            num_feat += 1
        if use_pop_feat:
            num_feat += 1
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        if use_feat_mixer:
            self.feat_mixer = FeatTokenMixer(hidden_size, num_feat, dropout)

        self.norm = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def get_attention_mask(self, mask_a, mask_b):
        """From CRAFT: creates attention mask of shape (B, 1, query_len, key_len)."""
        extended_attention_mask = jt.bmm(
            mask_a.unsqueeze(1).transpose(1, 2),
            mask_b.unsqueeze(1).float()
        ).bool().unsqueeze(1)
        extended_attention_mask = jt.where(extended_attention_mask, 0.0, -10000.0)
        return extended_attention_mask

    def _set_ui_tensors(self, ui_data):
        self.ui_head = jt.array(ui_data['head_list']).stop_grad()
        self.ui_tail = jt.array(ui_data['tail_list']).stop_grad()
        self.ui_weight = jt.array(ui_data['edge_weights']).stop_grad()

    def update_ui_graph(self, ui_data):
        if ui_data is not None and ui_data['num_edges'] > 0:
            self._set_ui_tensors(ui_data)
        else:
            self.ui_head = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                if m.weight.requires_grad:
                    m.weight = jt.array(np.random.normal(0, 0.02, m.weight.shape))
            elif isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    m.weight = jt.array(np.random.normal(0, 0.02, m.weight.shape))
                    if m.bias is not None:
                        m.bias = jt.zeros(m.bias.shape)
            elif isinstance(m, nn.LayerNorm):
                if m.bias is not None:
                    m.bias = jt.zeros(m.bias.shape)
                m.weight = jt.ones(m.weight.shape)

    def _ui_propagate(self, node_emb):
        """LightGCN-style UI graph propagation with symmetric normalization."""
        N_total = int(node_emb.shape[0])
        D = int(node_emb.shape[1])
        n_nodes = int(self.ui_n_nodes)
        emb = node_emb[:n_nodes]

        ego = emb
        layer_outs = [ego]

        for _ in range(self.ui_layers):
            head_emb = emb[self.ui_head]                           # (E, D)
            messages = head_emb * self.ui_weight.unsqueeze(-1)     # (E, D)

            aggregated = jt.zeros((n_nodes, D))
            aggregated = jt.scatter(aggregated, 0, self.ui_tail,
                                    messages, reduce='add')

            emb = aggregated  # pure neighbor aggregation, no residual
            layer_outs.append(emb)

        # LightGCN: mean of all layer outputs
        emb = sum(layer_outs) / len(layer_outs)

        if N_total > n_nodes:
            return jt.concat([emb, node_emb[n_nodes:]], dim=0)
        return emb

    def _ii_propagate(self, node_emb):
        """Propagate node embeddings through the II co-occurrence graph.

        Supports two modes:
        - static: fixed edge weights = co-occurrence counts (mean pooling).
        - dynamic attention: per-edge weights computed from
          ||head + user - tail||, softmax-normalized per tail.

        If ii_layer_weight is True, after each propagation layer the
        embeddings are scaled by their cosine similarity to the original
        (ego) embeddings, preventing over-smoothing (LayerGCN-style).
        """
        N_total = int(node_emb.shape[0])
        D = int(node_emb.shape[1])
        n_nodes = int(self.ii_n_nodes)
        emb = node_emb[:n_nodes]

        if self.ii_layer_weight:
            ego_emb = emb  # reference for anti-smoothing

        for _ in range(self.ii_layers):
            if self.ii_use_attn:
                # --- Dynamic attention: ||h + user - t|| ---
                h_emb = emb[self.ii_user_head]                       # (E, D)
                t_emb = emb[self.ii_user_tail]                       # (E, D)
                u_emb = node_emb[self.ii_user_list]                  # (E, D)

                # GTMSRec-style: min-max normalize distances then exp
                dist = jt.norm((h_emb + u_emb) - t_emb, dim=-1)    # (E,)
                d_min = dist.min()
                d_max = dist.max()
                norm_dist = (dist - d_min) / (d_max - d_min + 1e-8) # [0, 1]
                att_exp = jt.exp(norm_dist / self.ii_attn_tau)

                # Softmax per tail node
                tail_sum = jt.zeros((n_nodes,))
                tail_sum = jt.scatter(tail_sum, 0, self.ii_user_tail,
                                      att_exp, reduce='add')
                att_scores = att_exp / (tail_sum[self.ii_user_tail] + 1e-12)

                # Weighted aggregation (already mean-normalized by softmax)
                messages = h_emb * att_scores.unsqueeze(-1)           # (E, D)
                aggregated = jt.zeros((n_nodes, D))
                aggregated = jt.scatter(aggregated, 0, self.ii_user_tail,
                                        messages, reduce='add')
            else:
                # --- Static: fixed co-occurrence weights ---
                head_emb = emb[self.ii_head]                          # (E, D)
                messages = head_emb * self.ii_weight.unsqueeze(-1)    # (E, D)

                aggregated = jt.zeros((n_nodes, D))
                aggregated = jt.scatter(aggregated, 0, self.ii_tail,
                                        messages, reduce='add')
                aggregated = aggregated / self.ii_in_deg_clamped.unsqueeze(-1)

            if self.ii_layer_weight:
                # Cosine similarity: aggregated vs ego, then scale aggregated
                agg_norm = jt.norm(aggregated, dim=-1)
                ego_norm = jt.norm(ego_emb, dim=-1)
                dot = (aggregated * ego_emb).sum(dim=-1)
                _weights = dot / (agg_norm * ego_norm + 1e-8)
                aggregated = aggregated * _weights.unsqueeze(-1)

            emb = emb + aggregated

        if N_total > n_nodes:
            return jt.concat([emb, node_emb[n_nodes:]], dim=0)
        return emb

    def forward(
        self,
        src_neighb_seq,
        src_neighb_seq_len,
        neighbors_interact_times,
        cur_times,
        test_dst=None,
        dst_last_update_times=None,
        dst_neighb_seq=None,
        src_ids=None,
        global_step=None,
    ):
        """Forward pass.

        Args:
            src_neighb_seq: (B, L) src's historical neighbor IDs
            src_neighb_seq_len: (B,) valid neighbor counts
            neighbors_interact_times: (B, L) interaction times of neighbors
            cur_times: (B,) current prediction time
            test_dst: (B, 1+N) candidate dst IDs (col0=positive, rest=negatives)
            dst_last_update_times: (B, 1+N) last interaction time for each candidate
            dst_neighb_seq: (B, 1+N, K) candidate dsts' neighbor IDs (optional for CN)
            global_step: int or None, current training step (for neg augmentation)
        """
        B = src_neighb_seq.shape[0]
        L = src_neighb_seq.shape[1]
        T = test_dst.shape[1]  # 1 + num_neg

        # ---- Clamp negative indices to padding ----
        src_neighb_seq = jt.where(
            src_neighb_seq < 0, jt.zeros_like(src_neighb_seq), src_neighb_seq
        )
        if dst_neighb_seq is not None:
            dst_neighb_seq = jt.where(
                dst_neighb_seq < 0, jt.zeros_like(dst_neighb_seq), dst_neighb_seq
            )

        # ---- Node embeddings (UI ∥ II, parallel FREEDOM-style) ----
        raw_emb = self.node_embedding.weight
        n_nodes = max(self.ui_n_nodes if self.ui_head is not None else 0,
                      self.ii_n_nodes if self.ii_head is not None else 0,
                      int(raw_emb.shape[0]))
        ui_out = self._ui_propagate(raw_emb) if self.ui_head is not None else raw_emb[:n_nodes]
        ii_out = self._ii_propagate(raw_emb) if self.ii_head is not None else raw_emb[:n_nodes]
        if self.ui_head is not None and self.ii_head is not None:
            prop_emb = ui_out + ii_out - raw_emb[:n_nodes]
        elif self.ui_head is not None:
            prop_emb = ui_out
        elif self.ii_head is not None:
            prop_emb = ii_out
        else:
            prop_emb = raw_emb[:n_nodes]
        if prop_emb.shape[0] < raw_emb.shape[0]:
            prop_emb = jt.concat([prop_emb, raw_emb[prop_emb.shape[0]:]], dim=0)

        src_pos0_emb = ui_out[src_neighb_seq[:, 0:1]]        # (B, 1, D)
        neighb_emb = prop_emb[src_neighb_seq[:, 1:]]          # (B, L-1, D)
        src_neighb_emb = jt.cat([src_pos0_emb, neighb_emb], dim=1)  # (B, L, D)

        # CN uses neighbor-only (without src at position 0)
        raw_src_neighb_emb = neighb_emb
        neighb_seq_for_cn = src_neighb_seq[:, 1:]  # (B, L-1)

        test_dst_emb = prop_emb[test_dst]  # (B, T, D)

        pad_mask = src_neighb_seq != 0  # (B, L)

        # ---- Src embedding: from ui_out, reused for FiLM + feature tokens ----
        if src_ids is not None:
            src_emb = ui_out[src_ids].unsqueeze(1)  # (B, 1, D)
        else:
            src_emb = None

        # ---- Dst neighbor pooling: enrich candidate query with its own neighbors ----
        if self.use_dst_neighbor_pooling and dst_neighb_seq is not None:
            # Mask src_ids from dst neighbors BEFORE embedding lookup
            if src_ids is not None:
                src_mask = dst_neighb_seq != src_ids.numpy().reshape(-1, 1, 1)
                dst_neighb_seq = dst_neighb_seq * src_mask
            dst_neighb_emb = self.node_embedding(dst_neighb_seq)  # (B, T, K, D)
            K = dst_neighb_seq.shape[2]
            # Position-based exponential decay: recent=low position gets higher weight
            pos_decay = jt.array(np.exp(-np.arange(K, dtype=np.float32) / (K / 2.0)))  # (K,)
            pos_decay = pos_decay.reshape(1, 1, K, 1)
            # Mask out padding
            valid = (dst_neighb_seq != 0).float().unsqueeze(-1)  # (B, T, K, 1)
            weights = pos_decay * valid
            weights = weights / weights.sum(dim=2, keepdim=True).maximum(1.0)
            dst_neighbor_feat = (dst_neighb_emb * weights).sum(dim=2)  # (B, T, D)
            test_dst_emb = test_dst_emb + dst_neighbor_feat

        # Capture for InfoNCE (before norm)
        self._infonce_src = src_emb
        self._infonce_dst = test_dst_emb

        # ---- Norm + dropout ----
        src_neighb_emb = self.norm(src_neighb_emb.view(B * L, self.hidden_size)).view(B, L, self.hidden_size)
        src_neighb_emb = self.emb_dropout(src_neighb_emb)

        test_dst_emb = self.norm(test_dst_emb.view(B * T, self.hidden_size)).view(B, T, self.hidden_size)
        test_dst_emb = self.emb_dropout(test_dst_emb)

        # ---- Token mixer on src neighbor sequence ----
        if self.use_token_mixer:
            for layer in self.token_mixer:
                src_neighb_emb = layer(
                    src_neighb_emb,
                    key_padding_mask=(1 - pad_mask.int()))

        # ---- Inject time bucket features into Q, K, V before cross-attention ----
        # Key/Value: time since each neighbor interaction
        src_dt = (cur_times.view(-1, 1).float() - neighbors_interact_times.float())
        src_time_feat = self.time_bucket(src_dt.view(-1, 1)).view(B, L, self.hidden_size)
        src_neighb_emb = src_neighb_emb + src_time_feat * pad_mask.float().unsqueeze(-1)

        # Query: time since candidate's last interaction
        if dst_last_update_times is not None:
            dst_dt = (cur_times.view(-1, 1).float() - dst_last_update_times.float())
            dst_time_feat = self.time_bucket(dst_dt.view(-1, 1)).view(B, T, self.hidden_size)
            test_dst_emb = test_dst_emb + dst_time_feat

        # ---- Cross-attention: dst candidates attend to src neighbors ----
        attention_mask = src_neighb_seq != 0  # (B, L)
        test_dst_mask = jt.ones(test_dst_emb.shape[0], test_dst_emb.shape[1])
        extended_attention_mask = self.get_attention_mask(test_dst_mask, mask_b=attention_mask)
        attn_out = self.cross_attn(
            test_dst_emb, extended_attention_mask, key=src_neighb_emb
        )
        # attn_out: (B, T, D)

        repeat_feat = None
        if self.output_cat_repeat_times:
            # ---- Repeat count features ----
            repeat_times = test_dst.view(B, T, 1) == neighb_seq_for_cn.view(B, 1, L - 1)
            repeat_times = repeat_times.sum(dim=-1).unsqueeze(-1).float()  # (B, T, 1)
            repeat_feat = self.repeat_times_proj(repeat_times.view(-1, 1)).view(B, T, self.hidden_size)
            repeat_feat = self.repeat_times_norm(repeat_feat)
            repeat_feat = self.dropout(repeat_feat)

        # ---- NCN features (structural common neighbors) ----
        if dst_neighb_seq is not None:
            cn_features = self._compute_batch_cn(
                neighb_seq_for_cn, dst_neighb_seq, raw_src_neighb_emb, B, L - 1, T
            )
        else:
            cn_features = jt.zeros(B, T, self.hidden_size)

        # ---- Src identity + src-dst interaction features ----
        if src_ids is not None:
            src_emb_expand = src_emb.expand(B, T, self.hidden_size)  # (B, 1, D) → (B, T, D)
            src_xij = src_emb_expand * attn_out                      # (B, T, D)
        else:
            src_emb_expand = jt.zeros(B, T, self.hidden_size)
            src_xij = jt.zeros(B, T, self.hidden_size)

        # ---- Dst popularity feature (time-conditioned degree) ----
        if self.use_pop_feat:
            if dst_neighb_seq is not None:
                dst_deg = (dst_neighb_seq != 0).sum(dim=-1, keepdim=True).float()  # (B, T, 1)
            else:
                dst_deg = jt.zeros(B, T, 1)
            dst_pop = jt.log(dst_deg+1)
            dst_pop_feat = self.degree_proj(dst_pop.reshape(-1, 1)).view(B, T, self.hidden_size)
            dst_pop_feat = self.degree_norm(dst_pop_feat)
            dst_pop_feat = self.dropout(dst_pop_feat)

        # ---- Stack features as tokens, apply feature mixing, then MLP ----
        #feat_list = [attn_out, dst_time_feat, cn_features, src_emb_expand, src_xij]
        feat_list = [attn_out, cn_features, src_emb_expand, src_xij]
        if repeat_feat is not None:
            feat_list.insert(2, repeat_feat)  # after dst_time_feat, before cn_features
        if self.use_pop_feat:
            feat_list.append(dst_pop_feat)
        feat_stack = jt.stack(feat_list, dim=2)
        if self.use_feat_mixer:
            pooled = self.feat_mixer(feat_stack)  # (B, T, D)
        else:
            pooled = feat_stack.mean(dim=2)       # (B, T, D) simple mean pool
        output = self.output_layer(pooled.view(B * T, -1)).view(B, T, -1).squeeze(-1)
        return output

    def _compute_batch_cn(self, src_neighb_seq, dst_neighb_seq, x, B, L, T):
        """Compute per-pair common neighbor features.

        Uses batched dense local adjacency for efficiency.
        """
        K = dst_neighb_seq.shape[2]  # num dst neighbors

        # src neighbor sets: (B, L) IDs
        # dst neighbor sets: (B, T, K) IDs
        src_set = src_neighb_seq.unsqueeze(1)  # (B, 1, L)
        dst_set = dst_neighb_seq  # (B, T, K)

        # Count CNs: how many of src's neighbors match dst's neighbors
        src_exp = src_set.unsqueeze(2)  # (B, 1, 1, L)
        dst_exp = dst_set.unsqueeze(-1)  # (B, T, K, 1)

        cn_mask = (src_exp == dst_exp) & (src_exp != 0)  # (B, T, K, L)
        cn_mask = cn_mask.float()

        # Aggregate CN embeddings:
        # For each pair, get the embeddings of common neighbors
        cn_sum = (
            cn_mask.sum(dim=2)  # (B, T, L) — per neighbor position, how many cn matches
        ).clamp(min_v=0.0, max_v=1.0)  # binary: is this neighbor a CN?
        cn_sum = cn_sum / cn_sum.sum(dim=-1, keepdim=True).clamp(min_v=1.0)  # normalize

        x_src = x  # (B, L, D)
        cn_emb = jt.matmul(cn_sum, x_src)  # (B, T, D)
        return cn_emb

    def predict(self, src_neighb_seq, src_neighb_seq_len, neighbors_interact_times,
                cur_pred_times, test_dst, dst_last_update_times, dst_neighb_seq=None,
                src_ids=None, global_step=None):
        logits = self.forward(
            src_neighb_seq, src_neighb_seq_len, neighbors_interact_times,
            cur_pred_times, test_dst=test_dst,
            dst_last_update_times=dst_last_update_times,
            dst_neighb_seq=dst_neighb_seq,
            src_ids=src_ids,
            global_step=global_step,
        )
        return logits  # (B, T)

    def calculate_loss(self, src_neighb_seq, src_neighb_seq_len,
                       neighbors_interact_times, cur_pred_times, test_dst,
                       dst_last_update_times, dst_neighb_seq=None,
                       src_ids=None, global_step=None):
        bs = test_dst.shape[0]
        logits = self.predict(
            src_neighb_seq, src_neighb_seq_len, neighbors_interact_times,
            cur_pred_times, test_dst, dst_last_update_times, dst_neighb_seq,
            src_ids=src_ids,
            global_step=global_step
        )

        labels = jt.zeros(bs, dtype=jt.int32)
        loss = nn.CrossEntropyLoss()(logits / self.temperature, labels)

        # Optional InfoNCE auxiliary loss
        if self.use_infonce and self.lambda_infonce > 0 and self._infonce_src is not None:
            src_enc = self._infonce_src.squeeze(1)                     # (B, D)
            src_n = src_enc / (src_enc.norm(dim=-1, keepdim=True) + 1e-8)
            dst_n = self._infonce_dst / (self._infonce_dst.norm(dim=-1, keepdim=True) + 1e-8)
            pos_sim = (src_n * dst_n[:, 0, :]).sum(dim=-1) / self.infonce_temperature
            neg_sim = jt.matmul(
                src_n.unsqueeze(1), dst_n[:, 1:, :].transpose(-1, -2)
            ).squeeze(1) / self.infonce_temperature
            logits_nce = jt.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
            nce_loss = nn.CrossEntropyLoss()(logits_nce, jt.zeros(bs, dtype=jt.int32))
            loss = loss + self.lambda_infonce * nce_loss

        # L2 regularization: only on embeddings used in this batch
        if self.lambda_l2 > 0:
            involved_ids = jt.cat([
                src_neighb_seq.reshape(-1),
                test_dst.reshape(-1),
            ], dim=0)
            if dst_neighb_seq is not None:
                involved_ids = jt.cat([involved_ids, dst_neighb_seq.reshape(-1)], dim=0)
            if src_ids is not None:
                involved_ids = jt.cat([involved_ids, src_ids.reshape(-1)], dim=0)
            involved_np = np.unique(involved_ids.numpy())
            involved_np = involved_np[involved_np > 0]
            if len(involved_np) > 0:
                l2_loss = (self.node_embedding.weight[involved_np] ** 2).mean()
                loss = loss + self.lambda_l2 * l2_loss

        return loss, logits, labels