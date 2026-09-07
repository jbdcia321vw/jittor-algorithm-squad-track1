import os
import os.path as osp
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['JT_SYNC'] = '1'

import json
import jittor as jt
import numpy as np
import random
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from jittor_geometric.data import TemporalData
from jittor_geometric.dataloader.temporal_dataloader import TemporalDataLoader, get_neighbor_sampler
from model_tncn import TNCNModel
from ii_graph import build_ii_graph, print_graph_stats, IncrementalUIGraph
import argparse
import math

jt.flags.use_cuda = 1

class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing.

    step < warmup_steps:  lr = base_lr * step / warmup_steps
    step >= warmup_steps: cosine decay from base_lr to eta_min
    """

    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1)
            lr = self.eta_min + (self.base_lr - self.eta_min) * \
                 (1 + math.cos(math.pi * progress)) / 2
        self.optimizer.lr = lr


def _load_csv_np(path):
    arr = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def test_val(model, loader, full_neighbor_sampler, num_neighbors):
    model.eval()
    ap_list, auc_list, mrr_list = [], [], []
    for batch_data in tqdm(loader, ncols=120, desc='Validation'):
        src = jt.array(batch_data.src)
        dst = jt.array(batch_data.dst)
        t = jt.array(batch_data.t)
        neg_dst = jt.array(batch_data.neg_dst)

        src_neighb_seq, _, src_neighb_interact_times = (
            full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=src.numpy(), node_interact_times=t.numpy(),
                num_neighbors=num_neighbors - 1)
        )
        # Prepend src node itself at position 0
        src_neighb_seq = np.concatenate([src.numpy().reshape(-1, 1), src_neighb_seq], axis=1)
        src_neighb_interact_times = np.concatenate([t.numpy().reshape(-1, 1), src_neighb_interact_times], axis=1)
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)

        neg_count = len(neg_dst) // len(dst)
        pos_item = jt.Var(dst.numpy().reshape(-1, 1))
        neg_item = jt.Var(neg_dst.numpy().reshape(len(dst), neg_count))
        test_dst = jt.cat([pos_item, neg_item], dim=1)

        # Sample dst neighbors for CN features
        dst_neighb_seq, _, dst_neighb_times = (
            full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=test_dst.flatten().numpy(),
                node_interact_times=np.broadcast_to(
                    t.numpy()[:, np.newaxis],
                    (len(t), test_dst.shape[1])
                ).flatten(),
                num_neighbors=num_neighbors)
        )
        dst_neighb_seq = dst_neighb_seq.reshape(len(test_dst), test_dst.shape[1], -1)

        dst_neighb_times_3d = dst_neighb_times.reshape(len(test_dst), test_dst.shape[1], -1)
        dst_last_neighbor = dst_neighb_seq[:, :, 0]
        dst_last_update_time = dst_neighb_times_3d[:, :, 0]
        dst_last_update_time[dst_neighb_seq[:, :, 0] == 0] = -100000
        dst_last_update_time = jt.Var(dst_last_update_time)

        logits = model.predict(
            src_neighb_seq=jt.Var(src_neighb_seq),
            src_neighb_seq_len=jt.Var(neighbor_num),
            neighbors_interact_times=jt.Var(src_neighb_interact_times),
            cur_pred_times=jt.Var(t),
            test_dst=test_dst,
            dst_last_update_times=dst_last_update_time,
            dst_neighb_seq=jt.Var(dst_neighb_seq),
            src_ids=jt.Var(src.numpy()),
        )

        logits_np = logits.numpy()
        
        pos_score = logits_np[:, 0]
        neg_score = logits_np[:, 1:].flatten()

        y_true = np.concatenate([np.ones_like(pos_score), np.zeros_like(neg_score)])
        y_score = np.concatenate([pos_score, neg_score])
        ap_list.append(average_precision_score(y_true, y_score))
        auc_list.append(roc_auc_score(y_true, y_score))

        ranks = (logits_np >= logits_np[:, 0:1]).sum(axis=1)
        rr = 1.0 / ranks
        mrr_list.append(rr.mean())
    #print(logits_np[0])
    return {
        'AP': np.mean(ap_list), 'AUC': np.mean(auc_list),
        'MRR': np.mean(mrr_list)
    }


def train(model, optimizer, train_loader, val_loader, full_neighbor_sampler,
          num_neighbors, test_num_neighbors, num_epochs, save_path, dataset_name,
          early_stop_patience=10, scheduler=None,
          test_src=None, test_time=None, test_candidates=None, output_dir=None,
          args=None, start_epoch=0, inc_ui_graph=None, ui_rebuild_freq=1):
    best_mrr = 0
    patience_counter = 0

    global_step = 0
    for epoch in range(num_epochs):
        actual_epoch = start_epoch + epoch + 1

        if inc_ui_graph is not None:
            inc_ui_graph.reset()
            model.update_ui_graph(None)

        model.train()
        train_losses = []
        for batch_idx, batch_data in enumerate(
            tqdm(train_loader, ncols=120, desc=f'Epoch {actual_epoch}')):
            src = jt.array(batch_data.src)
            dst = jt.array(batch_data.dst)
            t = jt.array(batch_data.t)
            neg_dst = jt.array(batch_data.neg_dst)

            src_neighb_seq, _, src_neighb_interact_times = (
                full_neighbor_sampler.get_historical_neighbors_left(
                    node_ids=src.numpy(), node_interact_times=t.numpy(),
                    num_neighbors=num_neighbors - 1)
            )
            # Prepend src node itself at position 0
            src_np = src.numpy()
            t_np = t.numpy()
            src_neighb_seq = np.concatenate([src_np.reshape(-1, 1), src_neighb_seq], axis=1)
            src_neighb_interact_times = np.concatenate([t_np.reshape(-1, 1), src_neighb_interact_times], axis=1)
            neighbor_num = (src_neighb_seq != 0).sum(axis=1)

            neg_count = len(neg_dst) // len(dst)
            pos_item = jt.Var(dst.numpy().reshape(-1, 1))
            neg_item = jt.Var(neg_dst.numpy().reshape(len(dst), neg_count))
            test_dst = jt.cat([pos_item, neg_item], dim=1)

            # Sample dst neighbors for CN features
            dst_neighb_seq, _, dst_neighb_times = (
                full_neighbor_sampler.get_historical_neighbors_left(
                    node_ids=test_dst.flatten().numpy(),
                    node_interact_times=np.broadcast_to(
                        t.numpy()[:, np.newaxis],
                        (len(t), test_dst.shape[1])
                    ).flatten(),
                    num_neighbors=num_neighbors)
            )
            dst_neighb_seq = dst_neighb_seq.reshape(len(test_dst), test_dst.shape[1], -1)
            dst_neighb_times = dst_neighb_times.reshape(len(test_dst), test_dst.shape[1], -1)

            dst_last_neighbor = dst_neighb_seq[:, :, 0]
            dst_last_update_time = dst_neighb_times[:, :, 0]
            dst_last_update_time[dst_neighb_seq[:, :, 0] == 0] = -100000
            dst_last_update_time = jt.Var(dst_last_update_time)

            loss, _, _ = model.calculate_loss(
                src_neighb_seq=jt.Var(src_neighb_seq),
                src_neighb_seq_len=jt.Var(neighbor_num),
                neighbors_interact_times=jt.Var(src_neighb_interact_times),
                cur_pred_times=jt.Var(t),
                test_dst=test_dst,
                dst_last_update_times=dst_last_update_time,
                dst_neighb_seq=jt.Var(dst_neighb_seq),
                src_ids=jt.Var(src.numpy()),
                global_step=global_step,
            )
            global_step += 1

            optimizer.zero_grad()
            optimizer.step(loss)
            if scheduler is not None:
                scheduler.step()
            jt.sync_all()
            train_losses.append(loss.item())

            if inc_ui_graph is not None:
                inc_ui_graph.add_edges(src.numpy(), dst.numpy())
                if (batch_idx + 1) % ui_rebuild_freq == 0:
                    model.update_ui_graph(inc_ui_graph.build_ui_data())

        val_res = test_val(model, val_loader, full_neighbor_sampler, test_num_neighbors)
        current_lr = optimizer.lr
        print(f'Epoch {actual_epoch}, Train Loss: {np.mean(train_losses):.4f}, LR: {current_lr:.2e}')
        print(f'Epoch {actual_epoch}, Val: AP={val_res["AP"]:.4f} AUC={val_res["AUC"]:.4f} MRR={val_res["MRR"]:.4f}')

        current_mrr = val_res['MRR']
        is_best = current_mrr > best_mrr
        if is_best:
            best_mrr = current_mrr
            patience_counter = 0
        else:
            patience_counter += 1

        # ---- Save model + args in folder: {dataset}/{MRR}_{epoch} ----
        folder_name = f'{current_mrr:.4f}_epoch{actual_epoch}'
        model_dir = osp.join(save_path, dataset_name, folder_name)
        os.makedirs(model_dir, exist_ok=True)

        # Save model weights
        jt.save(model.state_dict(), osp.join(model_dir, 'model.pkl'))

        # Save args as JSON
        if args is not None:
            args_dict = vars(args).copy()
            args_dict['val_mrr'] = current_mrr
            args_dict['best_mrr'] = best_mrr
            args_dict['epoch'] = epoch + 1
            with open(osp.join(model_dir, 'args.json'), 'w') as f:
                json.dump(args_dict, f, indent=2, default=str)

        # ---- Run test and save results in same-named folder under saved_result/{dataset}/ ----
        if test_src is not None and output_dir is not None:
            result_dir = osp.join(output_dir, dataset_name, folder_name)
            if not osp.exists(result_dir):
                os.makedirs(result_dir, exist_ok=True)
                scores = test_competition(
                    model, test_src, test_time, test_candidates,
                    full_neighbor_sampler, test_num_neighbors, batch_size=200)
                output_file = osp.join(result_dir, f'{dataset_name}_result.csv')
                with open(output_file, 'w') as f:
                    for row in scores:
                        f.write(','.join([f'{p:.4f}' for p in row]) + '\n')
                print(f'  -> Test results saved to {result_dir}')

        status = 'BEST' if is_best else f'No improvement ({patience_counter}/{early_stop_patience})'
        print(f'  -> {status} | epoch={actual_epoch} MRR={current_mrr:.4f} best={best_mrr:.4f}')

        if patience_counter >= early_stop_patience:
            print(f'\nEarly stopping triggered after {actual_epoch} epochs!')
            print(f'Best validation MRR: {best_mrr:.4f}')
            break

    return best_mrr


def test_competition(model, test_src, test_time, test_candidates,
                     full_neighbor_sampler, num_neighbors, batch_size=200):
    model.eval()
    all_scores = []
    num_samples = len(test_src)
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), ncols=120, desc='Testing'):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, num_samples)

        batch_src = test_src[start:end]
        batch_time = test_time[start:end]
        batch_cand = test_candidates[start:end]

        src_neighb_seq, _, src_neighb_interact_times = (
            full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=batch_src, node_interact_times=batch_time,
                num_neighbors=num_neighbors - 1)
        )
        # Prepend src node itself at position 0
        src_neighb_seq = np.concatenate([batch_src.reshape(-1, 1), src_neighb_seq], axis=1)
        src_neighb_interact_times = np.concatenate([batch_time.reshape(-1, 1), src_neighb_interact_times], axis=1)
        neighbor_num = (src_neighb_seq != 0).sum(axis=1)

        test_dst = jt.Var(batch_cand)

        dst_neighb_seq, _, dst_neighb_times = (
            full_neighbor_sampler.get_historical_neighbors_left(
                node_ids=test_dst.flatten().numpy(),
                node_interact_times=np.broadcast_to(
                    batch_time[:, np.newaxis],
                    (len(batch_time), test_dst.shape[1])
                ).flatten(),
                num_neighbors=num_neighbors)
        )
        dst_neighb_seq = dst_neighb_seq.reshape(len(test_dst), test_dst.shape[1], -1)
        dst_neighb_times_3d = dst_neighb_times.reshape(len(test_dst), test_dst.shape[1], -1)
        dst_last_update_time = dst_neighb_times_3d[:, :, 0]
        dst_last_update_time[dst_neighb_seq[:, :, 0] == 0] = -100000
        dst_last_update_time = jt.Var(dst_last_update_time)

        logits = model.forward(
            jt.Var(src_neighb_seq), jt.Var(neighbor_num),
            jt.Var(src_neighb_interact_times), jt.Var(batch_time),
            test_dst=test_dst, dst_last_update_times=dst_last_update_time,
            dst_neighb_seq=jt.Var(dst_neighb_seq),
            src_ids=jt.Var(batch_src),
        )
        # Min-max normalize per row to [0,1], preserves ranking
        scores = logits
        s_min = scores.min(dim=1, keepdims=True)
        s_max = scores.max(dim=1, keepdims=True)
        scores = (scores - s_min) / (s_max - s_min + 1e-8)
        scores = scores.clamp(min_v=1e-7, max_v=1.0 - 1e-7)
        all_scores.append(scores.numpy())

    return np.vstack(all_scores)


# Main
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2026, help='Random seed for reproducibility')
parser.add_argument('--dataset', type=str, default='dataset1')
parser.add_argument('--data_dir', type=str, default='../data_A')
parser.add_argument('--save_dir', type=str, default='./saved_models')
parser.add_argument('--output_dir', type=str, default='./saved_result')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--early_stop', type=int, default=2)
parser.add_argument('--num_neighbors', type=int, default=128)
parser.add_argument('--test_num_neighbors', type=int, default=256)
parser.add_argument('--neg_sampling_ratio', type=float, default=99.0)
parser.add_argument('--val_ratio', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--lr_scheduler', type=str, default='cosine',
                    choices=['none', 'cosine'])
parser.add_argument('--lr_eta_min', type=float, default=0.00002)
parser.add_argument('--lr_warmup_steps', type=int, default=1,
                    help='0 = auto (1 epoch), >0 = explicit steps')
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--add_cyclic_time_feat', type=bool, default=True)
parser.add_argument('--temperature', type=float, default=1)
parser.add_argument('--use_dst_neighbor_pooling', type=bool, default=True)
parser.add_argument('--use_seq_self_attn', type=bool, default=True)
parser.add_argument('--seq_encoder_layers', type=int, default=1)
parser.add_argument('--use_feat_mixer', type=bool, default=True)
parser.add_argument('--use_pop_feat', type=bool, default=True)
parser.add_argument('--use_film', type=bool, default=False)
parser.add_argument('--use_infonce', type=bool, default=False)
parser.add_argument('--lambda_infonce', type=float, default=0.1)
parser.add_argument('--infonce_temperature', type=float, default=0.07)
parser.add_argument('--output_cat_repeat_times', type=bool, default=True)
parser.add_argument('--lambda_l2', type=float, default=0.1)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--ii_layers', type=int, default=3)
parser.add_argument('--ii_use_attn', type=bool, default=True)
parser.add_argument('--ii_attn_tau', type=float, default=3)
parser.add_argument('--ii_layer_weight', type=bool, default=False)
parser.add_argument('--ui_layers', type=int, default=0)
parser.add_argument('--checkpoint', type=str, default=None,
                    help='Path to checkpoint folder (contains model.pkl and args.json)')

args = parser.parse_args()

# ---- Resume from checkpoint ----
resume_epoch = 0
if args.checkpoint is not None:
    ckpt_dir = args.checkpoint
    print(f'Resuming from checkpoint: {ckpt_dir}')

    # Load saved args
    args_path = osp.join(ckpt_dir, 'args.json')
    if osp.exists(args_path):
        with open(args_path, 'r') as f:
            saved_args = json.load(f)
        # Override current args with saved values (keep checkpoint path and data dir)
        for k, v in saved_args.items():
            if k not in ('checkpoint', 'data_dir', 'save_dir', 'output_dir'):
                setattr(args, k, v)
        resume_epoch = saved_args.get('epoch', 0)
        print(f'  Loaded args, resuming from epoch {resume_epoch + 1}')
    else:
        print(f'  Warning: args.json not found in {ckpt_dir}, using current args')

if args.output_dir is None:
    args.output_dir = args.data_dir
if args.dataset == 'dataset1':
    args.output_cat_repeat_times = True
    args.seq_encoder_layers = 1
    args.add_cyclic_time_feat = False #-
    args.use_dst_neighbor_pooling = True #+
    args.use_seq_self_attn = True #-
    args.use_feat_mixer = True
    args.use_pop_feat = False #-
    args.use_infonce = True
    args.lambda_infonce = 0.1
    args.hidden_size = 128
    args.val_ratio = 0.01
    args.num_neighbors = 128
    args.test_num_neighbors = 256
    args.ii_layers = 5
    args.ii_use_attn = True
    args.ii_attn_tau = 2
    args.ii_layer_weight = False
    args.ui_layers = 2
    args.time_bucket_boundaries = np.unique(np.concatenate([
        [0, 60, 300, 600, 1800, 3600],
        np.logspace(np.log10(7200), np.log10(1.3e8), 58).astype(np.int64)
    ]))
elif args.dataset == 'dataset2':
    args.batch_size = 256
    args.lr = 0.0002
    args.lr_eta_min = 0.00002
    #过拟合严重，建议只训练一个epoch
    args.dropout = 0.2
    args.output_cat_repeat_times = False
    args.seq_encoder_layers = 1
    args.use_dst_neighbor_pooling = True # +
    args.use_seq_self_attn = True # =
    args.use_feat_mixer = True # +
    args.use_pop_feat = False # +
    args.use_infonce = False
    args.lambda_infonce = 0.02
    args.hidden_size = 64
    args.val_ratio = 0.01
    args.num_neighbors = 128
    args.test_num_neighbors = 256
    args.ii_layers = 3
    args.ii_use_attn = True
    args.ii_attn_tau = 1
    args.ii_layer_weight = False
    args.ui_layers = 2
    # ds2: many zero intervals, log-scale from 1h to 266M
    args.time_bucket_boundaries = np.unique(np.concatenate([
        [0, 60, 300, 600, 1800, 3600],
        np.logspace(np.log10(7200), np.log10(2.7e8), 58).astype(np.int64)
    ]))
print('=' * 80)
print(f'TNCN Competition - Dataset: {args.dataset}')
print(args)
print('=' * 80)

# Set random seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
jt.set_seed(args.seed)

# Load data
train_arr = _load_csv_np(f'{args.data_dir}/{args.dataset}/train.csv')
test_arr = _load_csv_np(f'{args.data_dir}/{args.dataset}/test.csv')

src_np = train_arr[:, 0].astype(np.int32)
dst_np = train_arr[:, 1].astype(np.int32)
t_np = train_arr[:, 2].astype(np.int32)
edge_ids_np = np.arange(len(train_arr), dtype=np.int32) + 1

test_src = test_arr[:, 0].astype(np.int32)
test_time = test_arr[:, 1].astype(np.int32)
test_candidates = test_arr[:, 2:].astype(np.int32)

print(f'Train+Val: {len(train_arr)}, Test: {len(test_arr)}')

num_total = len(train_arr)
num_val = int(num_total * args.val_ratio)
num_train = num_total - num_val

train_data = TemporalData(
    src=jt.Var(src_np[:num_train]), dst=jt.Var(dst_np[:num_train]),
    t=jt.Var(t_np[:num_train]), edge_ids=jt.Var(edge_ids_np[:num_train])
)
val_data = TemporalData(
    src=jt.Var(src_np[num_train:]), dst=jt.Var(dst_np[num_train:]),
    t=jt.Var(t_np[num_train:]), edge_ids=jt.Var(edge_ids_np[num_train:])
)
full_data = TemporalData(
    src=jt.Var(src_np), dst=jt.Var(dst_np),
    t=jt.Var(t_np), edge_ids=jt.Var(edge_ids_np)
)

train_loader = TemporalDataLoader(
    train_data, batch_size=args.batch_size,
    neg_sampling_ratio=args.neg_sampling_ratio
)
val_loader = TemporalDataLoader(
    val_data, batch_size=args.batch_size,
    neg_sampling_ratio=args.neg_sampling_ratio
)

full_neighbor_sampler = get_neighbor_sampler(full_data, 'recent', seed=1)

# Model
max_node = max(int(src_np.max()), int(dst_np.max()), int(test_candidates.max()))
node_size = max_node + 1
dst_min = min(int(dst_np.min()), int(test_candidates.min()))
src_min = int(src_np.min())

# Build UI (user-item) interaction graph
inc_ui_graph = IncrementalUIGraph(node_size) if args.ui_layers > 0 else None

# Build II (item-item) co-occurrence graph for dst embedding propagation
ii_graph = None
if args.ii_layers > 0:
    if args.dataset == 'dataset1':
        src_np = np.concatenate([src_np, dst_np])
        dst_np = np.concatenate([dst_np, src_np])
        t_np = np.concatenate([t_np, t_np])
    else:
        src_np = np.concatenate([src_np, dst_np])
        dst_np = np.concatenate([dst_np, src_np])
        t_np = np.concatenate([t_np, t_np])
    ii_graph = build_ii_graph(src_np, dst_np, t_np, n_dst=node_size,
                              return_user_edges=args.ii_use_attn)
    print_graph_stats(ii_graph)

print(f'Node size: {node_size}, Src min: {src_min}, Dst min: {dst_min}')

model = TNCNModel(
    n_nodes=node_size, hidden_size=args.hidden_size,
    n_layers=args.n_layers,
    n_heads=args.n_heads, dropout=args.dropout,
    output_cat_repeat_times=args.output_cat_repeat_times,
    use_dst_neighbor_pooling=args.use_dst_neighbor_pooling,
    use_seq_self_attn=args.use_seq_self_attn,
    seq_encoder_layers=args.seq_encoder_layers,
    use_feat_mixer=args.use_feat_mixer,
    use_pop_feat=args.use_pop_feat,
    use_infonce=args.use_infonce,
    lambda_infonce=args.lambda_infonce,
    infonce_temperature=args.infonce_temperature,
    temperature=args.temperature,
    lambda_l2=args.lambda_l2,
    ii_graph=ii_graph,
    ii_layers=args.ii_layers,
    ii_use_attn=args.ii_use_attn,
    ii_attn_tau=args.ii_attn_tau,
    ii_layer_weight=args.ii_layer_weight,
    ui_graph=None,
    ui_layers=args.ui_layers,
    time_bucket_boundaries=args.time_bucket_boundaries,
)
# Load checkpoint weights if resuming
if resume_epoch > 0:
    ckpt_model_path = osp.join(args.checkpoint, 'model.pkl')
    model.load_state_dict(jt.load(ckpt_model_path))
    print(f'  Loaded model weights from {ckpt_model_path}')

optimizer = jt.nn.Adam(list(model.parameters()), lr=args.lr)

remaining_epochs = args.epochs - resume_epoch
scheduler = None
if args.lr_scheduler == 'cosine':
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.lr_warmup_steps if args.lr_warmup_steps > 0 else len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps,
                                      total_steps=total_steps,
                                      eta_min=args.lr_eta_min)
    # Advance scheduler to match resumed state
    completed_steps = resume_epoch * len(train_loader)
    for _ in range(completed_steps):
        scheduler.step()
    print(f'Using WarmupCosineScheduler: warmup={warmup_steps}, '
          f'total={total_steps}, completed={completed_steps}')
    print(f'  Resumed LR: {optimizer.lr:.2e}')

save_path = args.save_dir
os.makedirs(save_path, exist_ok=True)

# Train
print(f'\nTraining for {remaining_epochs} epoch(s) (resumed from epoch {resume_epoch + 1}) '
      f'with early stopping (patience={args.early_stop})...')
best_mrr = train(
    model, optimizer, train_loader, val_loader,
    full_neighbor_sampler, args.num_neighbors, args.test_num_neighbors, remaining_epochs,
    save_path, args.dataset, args.early_stop, scheduler=scheduler,
    test_src=test_src, test_time=test_time, test_candidates=test_candidates,
    output_dir=args.output_dir, args=args, start_epoch=resume_epoch,
    inc_ui_graph=inc_ui_graph, ui_rebuild_freq=1,
)

print('\n' + '=' * 80)
print('DONE')
print('=' * 80)
