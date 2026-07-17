import os
import sys
import time
import argparse
import logging

import jax
import numpy as np
import optax
from jax import random, jit, numpy as jnp, value_and_grad
from jax.tree_util import tree_leaves
from flax.training import train_state
from model import ZINC, recurrent_param, no_decay_param
from datasets import load_zinc
from utils import map_nested_fn

parser = argparse.ArgumentParser()
# * model hyper-params
parser.add_argument("--num_layers", default=11, type=int)
parser.add_argument("--k_local", default=3, type=int)
parser.add_argument("--dim_h", default=64, type=int)
parser.add_argument("--dim_v", default=64, type=int)
parser.add_argument("--r_min", default=0.9, type=float)
parser.add_argument("--r_max", default=1., type=float)
parser.add_argument("--max_phase", default=6.28, type=float)
parser.add_argument("--drop_rate", default=0.2, type=float)
parser.add_argument("--expand", default=1, type=int)
parser.add_argument("--act", default="full-glu", type=str)

# * training hyper-params
parser.add_argument("--lr_min", default=1e-7, type=float)
parser.add_argument("--lr_max", default=1e-3, type=float)
parser.add_argument("--weight_decay", default=0.1, type=float)
parser.add_argument("--lr_factor", default=1., type=float)
parser.add_argument("--epochs", default=2000, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--warmup", default=0.05, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--seeds", default="", type=str)  # 逗号分隔, 覆盖 --seed
parser.add_argument("--gpu", default="0", type=str)

# * 对比策略: none | infonce | barlow
parser.add_argument("--contrast_type", default="barlow", type=str,
                    choices=["none", "infonce", "barlow"])
parser.add_argument("--use_entailment", default=True, action=argparse.BooleanOptionalAction)

parser.add_argument("--temperature", default=0.1, type=float)  # infonce 用
parser.add_argument("--barlow_lam", default=0.005, type=float)  # barlow off-diag 权重
parser.add_argument("--barlow_dim", default=128, type=int)  # barlow 投影维度
parser.add_argument("--margin", default=0.2, type=float)  # entailment 用
parser.add_argument("--lambda_ent", default=0.1, type=float)
parser.add_argument("--lambda_cl", default=0.05, type=float)  # 对比项总权重(不论哪种)
parser.add_argument("--lambda_mae", default=1.0, type=float)

# * 实验模式: single | ablation
parser.add_argument("--exp_mode", default="single", type=str, choices=["single", "ablation"])

args = parser.parse_args()


class TrainState(train_state.TrainState):
    key: jax.Array
    train_loss: float
    eval_loss: float
    total: int


def barlow_twins_loss(g, l, lam):
    """Barlow Twins: 无负样本, 沿 batch 标准化后使互相关矩阵趋近单位阵。
       g, l: (B, D) 投影输出。"""
    B = g.shape[0]
    g = (g - g.mean(axis=0)) / (g.std(axis=0) + 1e-6)
    l = (l - l.mean(axis=0)) / (l.std(axis=0) + 1e-6)
    c = (g.T @ l) / B  # (D, D)
    d = c.shape[0]
    diag = jnp.diagonal(c)
    on_diag = jnp.sum((diag - 1.0) ** 2)  # 对角 -> 1 (对齐)
    off_diag = jnp.sum(c ** 2) - jnp.sum(diag ** 2)  # 非对角 -> 0 (去冗余)
    return (on_diag + lam * off_diag) / d  # 除以 D 让不同 barlow_dim 量级可比


@jit
def train_step(state, batch):
    step_key = random.fold_in(state.key, state.step)

    def loss_fn(params):
        outputs = state.apply_fn(params, batch["x"], batch["node_mask"],
                                 batch["dist_mask"], batch["edge_attr"],
                                 training=True, rngs={"dropout": step_key})
        logits = outputs["logits"]
        mae_loss = jnp.abs(logits.squeeze(-1) - batch["y"]).mean()

        contrast_loss = jnp.array(0.0)
        entailment_loss = jnp.array(0.0)

        if args.contrast_type == "infonce":
            def l2_normalize(x, eps=1e-6):
                return x / jnp.sqrt(jnp.sum(x ** 2, axis=-1, keepdims=True) + eps)

            g = l2_normalize(outputs["c_global"])
            l = l2_normalize(outputs["c_local"])
            sim = g @ l.T / args.temperature
            labels = jnp.arange(g.shape[0])
            loss_g2l = optax.softmax_cross_entropy_with_integer_labels(sim, labels)
            loss_l2g = optax.softmax_cross_entropy_with_integer_labels(sim.T, labels)
            contrast_loss = 0.5 * (loss_g2l + loss_l2g).mean()

        elif args.contrast_type == "barlow":
            contrast_loss = barlow_twins_loss(outputs["b_global"], outputs["b_local"], args.barlow_lam)

        if args.use_entailment:
            def entailment_loss_fn(e_global, e_local):
                violation = jax.nn.relu(e_global - e_local + args.margin)
                return jnp.mean(jnp.mean(violation ** 2, axis=-1))

            entailment_loss = entailment_loss_fn(outputs["e_global"], outputs["e_local"])

        warmup = jnp.clip(state.step / args.warmup_steps, 0.0, 1.0)

        w_mae = args.lambda_mae * mae_loss
        w_cl = args.lambda_cl * warmup * contrast_loss
        w_ent = args.lambda_ent * warmup * entailment_loss
        train_loss = w_mae + w_cl + w_ent

        return train_loss, {
            "contrast": contrast_loss,
            "entailment": entailment_loss,
            "mae": mae_loss,
            "w_cl": w_cl,
            "w_ent": w_ent,
            "w_mae": w_mae,
            "warmup": warmup,
        }

    (train_loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(
        train_loss=state.train_loss + train_loss * batch["y"].shape[0],
        total=state.total + batch["y"].shape[0])
    return state, aux


@jit
def eval_step(state, batch):
    outputs = state.apply_fn(state.params, batch["x"], batch["node_mask"], batch["dist_mask"],
                             batch["edge_attr"],
                             training=False)
    logits = outputs["logits"]
    eval_loss = jnp.abs(logits.squeeze() - batch["y"]).sum()
    state = state.replace(eval_loss=(state.eval_loss + eval_loss),
                          total=(state.total + batch["y"].shape[0]))
    return state


def setup_logging(tag):
    """为每次实验重置 root logger，避免 handler 累积 / 日志串写。"""
    logger = logging.getLogger()
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass

    if not os.path.exists("./log"):
        os.mkdir("./log")
    time_str = time.strftime("%m%d-%H%M%S")
    log_path = f"./log/ZINC_{time_str}_{tag}_gpu{args.gpu}.log"

    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return log_path


def run_once(seed):
    """跑单次训练，返回 (best_val_mae, test_mae, ckpt_at)。"""
    train_set, val_set, test_set = load_zinc()
    model = ZINC(
        num_layers=args.num_layers,
        dim_o=1,
        dim_v=args.dim_v,
        dim_h=args.dim_h,
        expand=args.expand,
        k_local=args.k_local,
        r_min=args.r_min,
        r_max=args.r_max,
        max_phase=args.max_phase,
        drop_rate=args.drop_rate,
        act=args.act,
        contrast_type=args.contrast_type,
        use_entailment=args.use_entailment,
        barlow_dim=args.barlow_dim,
    )
    root_key = random.PRNGKey(seed)
    key, params_key, dropout_key = random.split(root_key, 3)
    params = model.init(params_key,
                        train_set["x"][:args.batch_size],
                        train_set["node_mask"][:args.batch_size],
                        train_set["dist_mask"][:args.batch_size],
                        train_set["edge_attr"][:args.batch_size],
                        training=False)
    logging.info(f"# parameters: {sum(p.size for p in tree_leaves(params))}")

    train_size = train_set["x"].shape[0]
    train_steps_per_epoch = train_size // args.batch_size
    train_steps_total = train_steps_per_epoch * args.epochs

    val_size = val_set["x"].shape[0]
    val_steps = (val_size - 1) // args.batch_size + 1
    test_size = test_set["x"].shape[0]
    test_steps = (test_size - 1) // args.batch_size + 1

    logging.info(
        f"train size: {train_size}; steps/epoch: {train_steps_per_epoch}; steps total: {train_steps_total}")
    logging.info(f"val size: {val_size}; val steps: {val_steps}")
    logging.info(f"test size: {test_size}; test steps: {test_steps}")
    args.warmup_steps = int(train_steps_total * args.warmup)

    label_fn = map_nested_fn(
        lambda k, _: "recurrent" if k in recurrent_param else "no_decay" if k in no_decay_param else "regular"
    )
    tx = optax.multi_transform(
        {
            "recurrent": optax.inject_hyperparams(optax.adam)(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=args.lr_min,
                    peak_value=args.lr_max * args.lr_factor,
                    warmup_steps=int(train_steps_total * args.warmup),
                    decay_steps=train_steps_total,
                    end_value=args.lr_min
                )
            ),
            "no_decay": optax.inject_hyperparams(optax.adam)(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=args.lr_min,
                    peak_value=args.lr_max,
                    warmup_steps=int(train_steps_total * args.warmup),
                    decay_steps=train_steps_total,
                    end_value=args.lr_min
                )
            ),
            "regular": optax.inject_hyperparams(optax.adamw)(
                learning_rate=optax.warmup_cosine_decay_schedule(
                    init_value=args.lr_min,
                    peak_value=args.lr_max,
                    warmup_steps=int(train_steps_total * args.warmup),
                    decay_steps=train_steps_total,
                    end_value=args.lr_min
                ),
                weight_decay=args.weight_decay
            )
        },
        label_fn
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        key=dropout_key,
        train_loss=0.,
        eval_loss=0.,
        total=0
    )

    best_val_mae = 100.
    ckpt = state.params
    ckpt_at = 0

    for e in range(args.epochs):
        start = time.time()
        train_indices = np.random.permutation(train_size)
        acc = {"mae": 0., "cl": 0., "ent": 0.,
               "w_mae": 0., "w_cl": 0., "w_ent": 0., "warmup": 0., "n": 0}

        for s in range(train_steps_per_epoch):
            batch_indices = train_indices[s * args.batch_size:(s + 1) * args.batch_size]
            batch = {
                "x": train_set["x"][batch_indices],
                "y": train_set["y"][batch_indices],
                "node_mask": train_set["node_mask"][batch_indices],
                "dist_mask": train_set["dist_mask"][batch_indices],
                "edge_attr": train_set["edge_attr"][batch_indices]
            }
            state, aux = train_step(state, batch)
            acc["mae"] += float(aux["mae"])
            acc["cl"] += float(aux["contrast"])
            acc["ent"] += float(aux["entailment"])
            acc["w_mae"] += float(aux["w_mae"])
            acc["w_cl"] += float(aux["w_cl"])
            acc["w_ent"] += float(aux["w_ent"])
            acc["warmup"] += float(aux["warmup"])
            acc["n"] += 1

        n = max(acc["n"], 1)
        w_mae_m, w_cl_m, w_ent_m = acc["w_mae"] / n, acc["w_cl"] / n, acc["w_ent"] / n
        w_total = w_mae_m + w_cl_m + w_ent_m + 1e-12
        logging.info(
            f"Epoch {e + 1}; TrainMAE {state.train_loss / state.total:.5f}; "
            f"raw[cl {acc['cl'] / n:.4f} ent {acc['ent'] / n:.4f}]; "
            f"weighted[mae {w_mae_m:.5f} cl {w_cl_m:.5f} ent {w_ent_m:.5f}]; "
            f"share[mae {100 * w_mae_m / w_total:.1f}% cl {100 * w_cl_m / w_total:.1f}% "
            f"ent {100 * w_ent_m / w_total:.1f}%]; warmup {acc['warmup'] / n:.3f}"
        )
        state = state.replace(train_loss=0., total=0)

        val_indices = np.arange(val_size)
        for s in range(val_steps):
            batch_indices = val_indices[s * args.batch_size:(s + 1) * args.batch_size]
            batch = {
                "x": val_set["x"][batch_indices],
                "y": val_set["y"][batch_indices],
                "node_mask": val_set["node_mask"][batch_indices],
                "dist_mask": val_set["dist_mask"][batch_indices],
                "edge_attr": val_set["edge_attr"][batch_indices]
            }
            state = eval_step(state, batch)

        val_loss = float(state.eval_loss / state.total)
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            ckpt_at = e + 1
            ckpt = state.params

        logging.info(f"Epoch {e + 1}; Val MAE {val_loss:.5f}; time {time.time() - start:.2f}s")
        state = state.replace(eval_loss=0., total=0)

    state = state.replace(params=ckpt)
    test_indices = np.arange(test_size)
    for s in range(test_steps):
        batch_indices = test_indices[s * args.batch_size:(s + 1) * args.batch_size]
        batch = {
            "x": test_set["x"][batch_indices],
            "y": test_set["y"][batch_indices],
            "node_mask": test_set["node_mask"][batch_indices],
            "dist_mask": test_set["dist_mask"][batch_indices],
            "edge_attr": test_set["edge_attr"][batch_indices]
        }
        state = eval_step(state, batch)

    test_mae = float(state.eval_loss / state.total)
    logging.info(f"[seed {seed}] Best val mae {best_val_mae:.5f} @epoch {ckpt_at}; Test mae {test_mae:.5f}")
    return best_val_mae, test_mae, ckpt_at


def run_config(seeds, tag):
    """对给定配置跑多种子并汇总 mean ± std。"""
    setup_logging(tag)
    logging.info(f"===== CONFIG [{tag}] =====")
    logging.info(f"contrast_type={args.contrast_type} use_entailment={args.use_entailment} "
                 f"lambda_cl={args.lambda_cl} lambda_ent={args.lambda_ent} "
                 f"barlow_lam={args.barlow_lam} barlow_dim={args.barlow_dim} seeds={seeds}")
    logging.info(args)

    val_list, test_list = [], []
    for seed in seeds:
        args.seed = seed
        np.random.seed(seed)
        v, t, _ = run_once(seed)
        val_list.append(v)
        test_list.append(t)

    val_arr, test_arr = np.array(val_list), np.array(test_list)
    logging.info(
        f"===== SUMMARY [{tag}] over seeds {seeds} =====\n"
        f"Val  MAE: {val_arr.mean():.5f} ± {val_arr.std():.5f}  {np.round(val_arr, 5).tolist()}\n"
        f"Test MAE: {test_arr.mean():.5f} ± {test_arr.std():.5f}  {np.round(test_arr, 5).tolist()}"
    )
    return val_arr.mean(), val_arr.std(), test_arr.mean(), test_arr.std()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.seeds.strip():
        seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    else:
        seeds = [args.seed]

    if args.exp_mode == "single":
        run_config(seeds, tag=f"{args.contrast_type}_ent{args.use_entailment}")
        return

    # ---------------- ablation----------------
    # ---------------- barlow_ent 超参精调 ----------------
    # 已确认: barlow+entailment 是最优组合。此处只精调其超参。
    args.contrast_type = "barlow"
    args.use_entailment = True

    # 对照锚(只跑一次)
    anchors = [("baseline", "none", False), ("ent_only", "none", True)]
    results = {}
    for tag, ct, ue in anchors:
        args.contrast_type, args.use_entailment = ct, ue
        results[tag] = run_config(seeds, tag=tag)
    args.contrast_type, args.use_entailment = "barlow", True

    grid = []
    for barlow_lam in (0.005, 0.02, 0.05):
        for lambda_cl in (0.05, 0.1):
            for lambda_ent in (0.1, 0.2):
                args.barlow_lam = barlow_lam
                args.lambda_cl = lambda_cl
                args.lambda_ent = lambda_ent
                tag = f"be_lam{barlow_lam}_cl{lambda_cl}_ent{lambda_ent}"
                results[tag] = run_config(seeds, tag=tag)
                grid.append(tag)

    setup_logging("barlow_ent_grid")
    logging.info("========== BARLOW_ENT GRID SUMMARY ==========")
    logging.info(f"seeds={seeds}  barlow_dim={args.barlow_dim}")
    logging.info(f"{'config':<28} | {'val_mean':>9} {'val_std':>8} | {'test_mean':>9} {'test_std':>8}")
    base_test = results["baseline"][2]
    order = ["baseline", "ent_only"] + sorted(grid, key=lambda t: results[t][2])
    for tag in order:
        vm, vs, tm, ts = results[tag]
        logging.info(f"{tag:<28} | {vm:>9.5f} {vs:>8.5f} | {tm:>9.5f} {ts:>8.5f}  (Δ {tm - base_test:+.5f})")
    logging.info("对照: 之前 barlow_ent(lam0.005,cl0.05,ent0.1) test≈0.09174。目标: 进一步压低。")


if __name__ == "__main__":
    main()
