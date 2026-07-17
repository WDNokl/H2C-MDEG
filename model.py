import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import lecun_normal, variance_scaling, normal
from init import init_eig_magnitude, init_eig_phase, init_gamma_log
from utils import binary_operator_diag

recurrent_param = ["B_re", "B_im", "nu_log", "theta_log", "gamma_log"]
no_decay_param = ["embedding", "scale", "bias"]

# From https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py#L78
full_atom_feature_dims = [119, 5, 12, 12, 10, 6, 6, 2, 2]
full_bond_feature_dims = [5, 6, 2]


class MLP(nn.Module):
    dim_h: int
    expand: int = 1
    drop_rate: float = 0.

    @nn.compact
    def __call__(self, inputs, training: bool = False):
        x = nn.LayerNorm()(inputs)
        x = nn.Dense(self.expand * self.dim_h)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.drop_rate, deterministic=not training)(x)
        x = nn.Dense(self.dim_h)(x)
        x = nn.Dropout(self.drop_rate, deterministic=not training)(x)
        return x + inputs


class LRU(nn.Module):
    dim_v: int
    dim_h: int

    r_min: float = 0.
    r_max: float = 1.
    max_phase: float = 6.28
    drop_rate: float = 0.
    act: str = "full-glu"

    @nn.compact
    def __call__(self, inputs, training: bool = False):

        xs = nn.LayerNorm()(inputs)

        nu_log = self.param("nu_log", init_eig_magnitude(self.r_min, self.r_max), (self.dim_v,))
        theta_log = self.param("theta_log", init_eig_phase(self.max_phase), (self.dim_v,))
        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        gamma_log = self.param("gamma_log", init_gamma_log, diag_lambda)

        B_re = self.param("B_re", variance_scaling(0.5, "fan_in", "truncated_normal"), (self.dim_h, self.dim_v))
        B_im = self.param("B_im", variance_scaling(0.5, "fan_in", "truncated_normal"), (self.dim_h, self.dim_v))
        B = B_re + 1j * B_im
        Bu = xs @ (B * jnp.exp(gamma_log))

        lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[0], axis=0)
        lambda_elements = jnp.expand_dims(lambda_elements, axis=(1, 2))
        _, xs = jax.lax.associative_scan(binary_operator_diag, (lambda_elements, Bu), reverse=True)
        x = xs[0]

        C_re = self.param("C_re", lecun_normal(), (self.dim_v, self.dim_h))
        C_im = self.param("C_im", lecun_normal(), (self.dim_v, self.dim_h))
        C = C_re + 1j * C_im

        x = nn.gelu((x @ C).real)
        x = nn.Dropout(self.drop_rate, deterministic=not training)(x)
        if self.act == "full-glu":
            x = nn.Dense(self.dim_h)(x) * jax.nn.sigmoid(nn.Dense(self.dim_h)(x))
        elif self.act == "half-glu":
            x = x * jax.nn.sigmoid(nn.Dense(self.dim_h)(x))
        x = nn.Dropout(self.drop_rate, deterministic=not training)(x)
        return x + inputs[0]


class GRED(nn.Module):
    dim_v: int
    dim_h: int
    expand: int = 1

    r_min: float = 0.
    r_max: float = 1.
    max_phase: float = 6.28
    drop_rate: float = 0.
    act: str = "full-glu"

    @nn.compact
    def __call__(self, inputs_global, dist_masks_global, inputs_local, dist_masks_local, training: bool = False):
        # global
        xs_global = jnp.swapaxes(dist_masks_global, 0, 1) @ inputs_global
        xs_global = MLP(self.dim_h, self.expand, self.drop_rate)(xs_global)
        z_global = LRU(
            self.dim_v,
            self.dim_h,
            self.r_min,
            self.r_max,
            self.max_phase,
            self.drop_rate,
            self.act
        )(xs_global, training=training)

        # local
        xs_local = jnp.swapaxes(dist_masks_local, 0, 1) @ inputs_local
        xs_local = MLP(self.dim_h, self.expand, self.drop_rate)(xs_local)
        z_local = LRU(
            self.dim_v,
            self.dim_h,
            self.r_min,
            self.r_max,
            self.max_phase,
            self.drop_rate,
            self.act
        )(xs_local, training=training)

        return z_global, z_local


class ZINC(nn.Module):
    num_layers: int
    dim_o: int

    dim_v: int
    dim_h: int
    expand: int = 1
    k_local: int = 1

    r_min: float = 0.
    r_max: float = 1.
    max_phase: float = 2 * jnp.pi
    drop_rate: float = 0.
    act: str = "full-glu"

    # 对比策略: "none" | "infonce" | "barlow"
    contrast_type: str = "barlow"
    use_entailment: bool = True
    # Barlow 投影维度 (通常比 dim_h 大, 这里给个可配置的倍数)
    barlow_dim: int = 128

    @nn.compact
    def __call__(self, inputs, node_masks, dist_masks, edge_attr=None, training: bool = False):
        x = nn.Embed(28, self.dim_h, embedding_init=normal(stddev=0.01))(inputs)  # 特征嵌入层
        x = nn.Dense(self.dim_h)(nn.gelu(x))  # 线性变换层

        e = nn.Embed(4, self.dim_h, embedding_init=normal(stddev=0.01))(edge_attr)  # 边嵌入层
        e = nn.Dense(self.dim_h)(nn.gelu(e))  # 线性变换层

        # global
        dist_masks_global = dist_masks
        deg_global = jnp.sum(dist_masks_global[:, 1], axis=-1, keepdims=True)
        deg_inv_global = jnp.where(deg_global > 0, 1 / deg_global, 0)  # 度归一化

        # 每个节点接收来自所有邻居的边特征，求平均后以残差方式更新自身特征。
        x_global = x + jnp.sum(dist_masks_global[:, 1, ..., None] * e, axis=-2) * deg_inv_global

        # local
        dist_masks_local = dist_masks[:, :self.k_local]
        deg_local = jnp.sum(dist_masks_local[:, 1], axis=-1, keepdims=True)
        deg_inv_local = jnp.where(deg_local > 0, 1 / deg_local, 0)

        x_local = x + jnp.sum(dist_masks_local[:, 1, ..., None] * e, axis=-2) * deg_inv_local

        for _ in range(self.num_layers):
            x_global, x_local = GRED(
                self.dim_v,
                self.dim_h,
                self.expand,
                self.r_min,
                self.r_max,
                self.max_phase,
                self.drop_rate,
                self.act
            )(x_global, dist_masks_global, x_local, dist_masks_local, training=training)

        # ---------- 门控融合 (对称插值) ----------
        gate = jax.nn.sigmoid(nn.Dense(self.dim_h)(jnp.concatenate([x_global, x_local], axis=-1)))
        x = gate * x_global + (1.0 - gate) * x_local  # (B, N, H)

        # ---------- 分类 / 回归头 (mean pooling) ----------
        x = jnp.where(jnp.expand_dims(node_masks, -1), x, 0.)
        denom = jnp.clip(jnp.sum(node_masks, axis=1, keepdims=True), a_min=1)
        x = jnp.sum(x, axis=1) / denom
        x = nn.gelu(nn.Dense(self.dim_h)(x))
        logits = nn.Dense(self.dim_o)(x)

        use_contrastive = self.contrast_type in ("infonce", "barlow")

        if use_contrastive or self.use_entailment:
            def masked_mean(z):
                z = jnp.where(jnp.expand_dims(node_masks, -1), z, 0.)
                denom = jnp.clip(jnp.sum(node_masks, axis=1, keepdims=True), a_min=1)
                return jnp.sum(z, axis=1) / denom  # (B, H)

            g_pool = masked_mean(x_global)
            l_pool = masked_mean(x_local)
            out = {"logits": logits}

            if self.contrast_type == "infonce":
                # 实例判别投影头 (归一化后算 cosine, 见 train)
                out["c_global"] = nn.Dense(self.dim_h)(nn.gelu(nn.Dense(self.dim_h)(g_pool)))
                out["c_local"] = nn.Dense(self.dim_h)(nn.gelu(nn.Dense(self.dim_h)(l_pool)))

            elif self.contrast_type == "barlow":
                # Barlow Twins 投影头: 更宽, 末层线性 (无 gelu), 不做 L2 归一化
                out["b_global"] = nn.Dense(self.barlow_dim)(nn.gelu(nn.Dense(self.barlow_dim)(g_pool)))
                out["b_local"] = nn.Dense(self.barlow_dim)(nn.gelu(nn.Dense(self.barlow_dim)(l_pool)))

            if self.use_entailment:
                out["e_global"] = nn.softplus(nn.Dense(self.dim_h)(nn.gelu(nn.Dense(self.dim_h)(g_pool))))
                out["e_local"] = nn.softplus(nn.Dense(self.dim_h)(nn.gelu(nn.Dense(self.dim_h)(l_pool))))
            return out
        else:
            return {"logits": logits}


class Peptides(nn.Module):
    num_layers: int
    dim_o: int

    dim_v: int
    dim_h: int
    expand: int = 1
    k_local: int = 1

    r_min: float = 0.
    r_max: float = 1.
    max_phase: float = 6.28
    drop_rate: float = 0.
    act: str = "full-glu"

    @nn.compact
    def __call__(self, inputs, node_masks, dist_masks, training: bool = False, stage='regression',
                 freeze_backbone: bool = False):
        x = 0
        for i in range(inputs.shape[-1]):
            x = x + nn.Embed(full_atom_feature_dims[i], self.dim_h, embedding_init=normal(stddev=0.01))(inputs[..., i])
        x = nn.Dense(self.dim_h)(nn.gelu(x))

        x_global = x
        x_local = x
        for _ in range(self.num_layers):
            x_global, x_local = GRED(
                self.dim_v,
                self.dim_h,
                self.expand,
                self.r_min,
                self.r_max,
                self.max_phase,
                self.drop_rate,
                self.act
            )(x_global, x_local, dist_masks, training=training)

        x_local_masked = jnp.where(jnp.expand_dims(node_masks, -1), x_local, 0.)
        z_local = jnp.sum(x_local_masked, axis=1)

        x_global_masked = jnp.where(jnp.expand_dims(node_masks, -1), x_global, 0.)
        # x_global = jnp.sum(x_global, axis=1)

        attn_scores = jnp.einsum('bd,bnd->bn', z_local, x_global_masked) / jnp.sqrt(self.dim_h)
        attn_scores = jnp.where(node_masks, attn_scores, -1e9)
        attn_weights = jax.nn.softmax(attn_scores, axis=1)
        z_global = jnp.einsum('bn,bnd->bd', attn_weights, x_global_masked)

        if freeze_backbone:
            z_global_reg = jax.lax.stop_gradient(z_global)
            z_local_reg = jax.lax.stop_gradient(z_local)
        else:
            z_global_reg = z_global
            z_local_reg = z_local

        proj_global = nn.Dense(self.dim_h)(z_global)
        proj_global = nn.gelu(proj_global)
        proj_global = nn.Dense(self.dim_h)(proj_global)

        proj_local = nn.Dense(self.dim_h)(z_local)
        proj_local = nn.gelu(proj_local)
        proj_local = nn.Dense(self.dim_h)(proj_local)

        context_gate_logits = nn.Dense(self.dim_h)(jnp.concatenate([z_global_reg, z_local_reg], axis=-1))
        context_gate = jax.nn.sigmoid(context_gate_logits)

        x_fused = z_local_reg + context_gate * z_global_reg
        x_fused = nn.LayerNorm()(x_fused)
        x_final_class = nn.Dense(self.dim_o)(x_fused)

        if stage == 'contrastive':
            return proj_global, proj_local

        elif stage == 'regression':
            return x_final_class
        else:
            raise ValueError(f"Unknown stage: {stage}")


class SuperPixel(nn.Module):
    num_layers: int
    dim_o: int

    dim_v: int
    dim_h: int
    expand: int = 1

    r_min: float = 0.
    r_max: float = 1.
    max_phase: float = 6.28
    drop_rate: float = 0.
    act: str = "full-glu"

    @nn.compact
    def __call__(self, inputs, node_masks, dist_masks, training: bool = False):
        x = nn.Dense(self.dim_h)(inputs)
        x = nn.Dense(self.dim_h)(nn.gelu(x))
        for _ in range(self.num_layers):
            x = GRED(
                self.dim_v,
                self.dim_h,
                self.expand,
                self.r_min,
                self.r_max,
                self.max_phase,
                self.drop_rate,
                self.act
            )(x, dist_masks, training=training)
        x = jnp.where(jnp.expand_dims(node_masks, -1), x, 0.)
        x = jnp.sum(x, axis=1) / jnp.sum(node_masks, axis=1, keepdims=True)
        x = nn.Dense(self.dim_o)(x)
        return x


class SBM(nn.Module):
    num_layers: int
    dim_o: int

    dim_v: int
    dim_h: int
    expand: int = 1

    r_min: float = 0.
    r_max: float = 1.
    max_phase: float = 6.28
    drop_rate: float = 0.
    act: str = "full-glu"

    @nn.compact
    def __call__(self, inputs, dist_masks, training: bool = False):
        x = nn.Embed(7, self.dim_h, embedding_init=normal(stddev=0.01))(inputs.argmax(axis=-1))
        for _ in range(self.num_layers):
            x = GRED(
                self.dim_v,
                self.dim_h,
                self.expand,
                self.r_min,
                self.r_max,
                self.max_phase,
                self.drop_rate,
                self.act
            )(x, dist_masks, training=training)
        x = nn.Dense(self.dim_o)(x)
        return x