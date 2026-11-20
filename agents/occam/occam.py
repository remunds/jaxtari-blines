"""OCCAM (Object-Centric Attention via Masking) for JAXtari.

Port of Blüml et al. (2025), arXiv:2504.03024, reference code
https://github.com/VanillaWhey/OCAtariWrappers. Masks are built from each game's
ObjectObservation groups (one group = one class), drawn at native resolution and
area-downscaled to 84x84, so edge pixels keep partial coverage values.

Usage: any pixel-based agent, with OCCAMWrapper in place of PixelObsWrapper
    uv run main.py +alg=dqn_rgb_tuned OCCAM_MASK=binary
    uv run main.py +alg=ppo_rgb ENV_ID=breakout OCCAM_MASK=planes

Mask modes (F = frame stack, K = object groups)
    object   grayscale pixels inside boxes, background 0    (F, 84, 84, 1)
    binary   255 inside any box                             (F, 84, 84, 1)
    class    one gray level per group                       (F, 84, 84, 1)
    planes   one binary plane per group                     (F * K, 84, 84, 1)
    opc      planes plus the grayscale frame as last channel (F * (K + 1), 84, 84, 1)

The grayscale channel of opc is the PixelObsWrapper frame: with native_downscaling (the
agents pass NATIVE_DOWNSCALING, default True) it comes from the game's native 84x84 grayscale
renderer, which needs `game_name` and `mods`; otherwise the frame is area-downscaled like the masks.

Python
    OCCAMWrapper(AtariWrapper(jaxtari.make("pong")), mask_mode="class")
    log_occam_comparison_video   eval clip [clean | boxes | mask] to W&B

A game may declare OCCAM_RENDER_ORDER (group names, back to front) for the visualization.
"""

from __future__ import annotations

import functools
import colorsys
from typing import Any, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct

from jaxtari.wrappers import (
    AtariWrapper,
    JaxtariWrapper,
    _bcast_where,
    _reset_with_active,
    _unroll_frame_skip,
    speculative_cond,
)
from jaxtari.environment import ObjectObservation
from jaxtari import spaces


MASK_MODES = ("object", "binary", "class", "planes", "opc")

_GRAY_W = jnp.array([0.2989, 0.5870, 0.1140], dtype=jnp.float32)


def _rgb_to_gray(frame_rgb: jnp.ndarray) -> jnp.ndarray:
    """(H, W, 3) -> (H, W) float32 grayscale, same weights as PixelObsWrapper."""
    return jnp.dot(frame_rgb.astype(jnp.float32), _GRAY_W)


def _area_weights(src: int, dst: int) -> np.ndarray:
    """(dst, src) area-resampling weights, equal to cv2.INTER_AREA (downscale only)."""
    assert src >= dst, f"_area_weights is downscale-only, got src={src} < dst={dst}"
    s = src / dst
    edges = np.arange(dst + 1, dtype=np.float64) * s
    lo = np.arange(src, dtype=np.float64)[None, :]
    overlap = np.minimum(lo + 1.0, edges[1:, None]) - np.maximum(lo, edges[:-1, None])
    return np.clip(overlap, 0.0, None) / s


def _area_resize(img: jnp.ndarray, w_y: jnp.ndarray, w_x: jnp.ndarray) -> jnp.ndarray:
    """Area-downscale the last two axes of `img` via `w_y @ img @ w_x`."""
    return jnp.matmul(jnp.matmul(w_y, img.astype(jnp.float32)), w_x)


def _make_gray_palette(n_classes: int) -> np.ndarray:
    """(K + 1,) uint8 gray levels (i + 1) * (255 // K), index 0 = background, as in the reference."""
    shade = 255 // max(n_classes, 1)
    levels = [(i + 1) * shade for i in range(max(n_classes, 1))]
    return np.array([0] + levels, dtype=np.uint8)


def _make_color_palette(n_classes: int) -> np.ndarray:
    """(K + 1, 3) uint8 RGB palette for visualization, index 0 = black."""
    cols = [(0, 0, 0)]
    for i in range(max(n_classes, 1)):
        h = i / max(n_classes, 1)
        r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
        cols.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(cols, dtype=np.uint8)


def _planes_isometric_rgb(
    planes: jnp.ndarray,
    palette: jnp.ndarray,
    out_h: int,
    out_w: int,
    shear: float = 0.25,
    gap_frac: float = 0.38,
    rise_frac: float = 0.05,
    bg_alpha: float = 0.20,
    aspect: float | None = None,
) -> jnp.ndarray:
    """Draw (K, h, w) planes as stacked translucent sheets like Fig. 3e; visualization only."""
    K, h, w = planes.shape
    A = (w / h) if aspect is None else float(aspect)

    ph_from_w = out_w / (A * (1.0 + (K - 1) * gap_frac))
    ph_from_h = out_h / (1.0 + (K - 1) * rise_frac + shear * A)
    ph = 0.94 * min(ph_from_w, ph_from_h)
    pw = A * ph
    sx, sy = pw / w, ph / h
    dx = gap_frac * pw
    dy = -rise_frac * ph

    need_w = pw + (K - 1) * dx
    need_h = ph + shear * pw - (K - 1) * dy
    ox = (out_w - need_w) / 2.0
    oy = -(K - 1) * dy + (out_h - need_h) / 2.0

    X = jnp.arange(out_w, dtype=jnp.float32)[None, :]
    Y = jnp.arange(out_h, dtype=jnp.float32)[:, None]

    canvas = jnp.zeros((out_h, out_w, 3), jnp.float32)
    for k in range(K - 1, -1, -1):
        xr = X - ox - k * dx
        u = xr / sx
        v = (Y - oy - k * dy - shear * xr) / sy
        inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ui = jnp.clip(u, 0, w - 1).astype(jnp.int32)
        vi = jnp.clip(v, 0, h - 1).astype(jnp.int32)

        hit = (planes[k][vi, ui] > 0) & inside
        eu, ev = 1.5 / sx, 1.5 / sy
        edge = inside & ((u < eu) | (u > w - 1 - eu) | (v < ev) | (v > h - 1 - ev))

        cls = palette[k + 1].astype(jnp.float32)
        col = jnp.where((hit | edge)[..., None], cls, jnp.float32(18.0))
        alpha = jnp.where(hit, 1.0, jnp.where(edge, 0.75, jnp.where(inside, bg_alpha, 0.0)))
        canvas = canvas * (1.0 - alpha[..., None]) + col * alpha[..., None]

    return jnp.clip(canvas, 0.0, 255.0).astype(jnp.uint8)


def _extract_object_groups(obs: Any) -> List[ObjectObservation]:
    """ObjectObservation leaves from obs PyTree, in deterministic PyTree order."""
    leaves = jax.tree_util.tree_leaves(
        obs, is_leaf=lambda n: isinstance(n, ObjectObservation)
    )
    return [leaf for leaf in leaves if isinstance(leaf, ObjectObservation)]


def _group_names(obs: Any) -> List[str]:
    """Names of the ObjectObservation leaves, in the same order as _extract_object_groups."""
    paths, _ = jax.tree_util.tree_flatten_with_path(
        obs, is_leaf=lambda n: isinstance(n, ObjectObservation)
    )
    names = []
    for path, leaf in paths:
        if isinstance(leaf, ObjectObservation):
            names.append(".".join(
                str(getattr(k, "name", getattr(k, "key", k))) for k in path
            ))
    return names


def _render_order(base_env: Any, names: List[str]) -> List[int]:
    """Back-to-front group indices from the game's OCCAM_RENDER_ORDER, else PyTree order."""
    declared = getattr(base_env, "OCCAM_RENDER_ORDER", None)
    if not declared:
        return list(range(len(names)))
    pos = {n: i for i, n in enumerate(names)}
    order = [pos[n] for n in declared if n in pos]
    seen = set(order)
    return order + [i for i in range(len(names)) if i not in seen]


def _sheets_width(base_w: int, k: int, gap_frac: float = 0.38) -> int:
    """Panel width the sheet stack needs for k planes."""
    return int(round(base_w * (1.0 + gap_frac * max(0, k - 1))))


def _group_box_arrays(group: ObjectObservation, img_h: int, img_w: int):
    """Normalize one ObjectObservation to (x, y, w, h, valid) arrays of shape (n,)."""
    x = jnp.atleast_1d(group.x).astype(jnp.int32)
    y = jnp.atleast_1d(group.y).astype(jnp.int32)
    w = jnp.atleast_1d(group.width).astype(jnp.int32)
    h = jnp.atleast_1d(group.height).astype(jnp.int32)

    n = x.shape[0]
    active = jnp.broadcast_to(jnp.atleast_1d(group.active).astype(bool), (n,))

    valid = (
        active
        & (w > 0)
        & (h > 0)
        & (x < img_w)
        & (y < img_h)
        & (x + w > 0)
        & (y + h > 0)
    )
    return x, y, w, h, valid


def _rasterize_group(x, y, w, h, valid, img_h: int, img_w: int) -> jnp.ndarray:
    """(H, W) bool occupancy of all valid boxes via a 2D difference array, O(n + H * W)."""
    x0 = jnp.clip(x, 0, img_w)
    x1 = jnp.clip(x + w, 0, img_w)
    y0 = jnp.clip(y, 0, img_h)
    y1 = jnp.clip(y + h, 0, img_h)
    inc = valid.astype(jnp.int32)

    diff = jnp.zeros((img_h + 1, img_w + 1), dtype=jnp.int32)
    diff = diff.at[y0, x0].add(inc)
    diff = diff.at[y0, x1].add(-inc)
    diff = diff.at[y1, x0].add(-inc)
    diff = diff.at[y1, x1].add(inc)

    counts = jnp.cumsum(jnp.cumsum(diff, axis=0), axis=1)
    return counts[:img_h, :img_w] > 0


def _union(group_masks: List[jnp.ndarray]) -> jnp.ndarray:
    """Logical OR over a non-empty list of (H, W) boolean masks."""
    union = group_masks[0]
    for gm in group_masks[1:]:
        union = union | gm
    return union


def _group_masks(obs: Any, img_h: int, img_w: int) -> List[jnp.ndarray]:
    """One (H, W) boolean occupancy mask per ObjectObservation group."""
    return [
        _rasterize_group(*_group_box_arrays(g, img_h, img_w), img_h, img_w)
        for g in _extract_object_groups(obs)
    ]


def _group_outlines(obs: Any, img_h: int, img_w: int) -> List[jnp.ndarray]:
    """One (H, W) boolean 1px box outline per ObjectObservation group."""
    outs = []
    for g in _extract_object_groups(obs):
        x, y, w, h, valid = _group_box_arrays(g, img_h, img_w)
        full = _rasterize_group(x, y, w, h, valid, img_h, img_w)
        inner = _rasterize_group(x + 1, y + 1, w - 2, h - 2, valid, img_h, img_w)
        outs.append(full & ~inner)
    return outs


def _class_map(group_masks: List[jnp.ndarray], order: List[int] | None = None) -> jnp.ndarray:
    """(H, W) int32 class ids 1..K, 0 = background; later groups in `order` win on overlap."""
    idx = range(len(group_masks)) if order is None else order
    class_map = jnp.zeros(group_masks[0].shape, dtype=jnp.int32)
    for k in idx:
        class_map = jnp.where(group_masks[k], k + 1, class_map)
    return class_map


@struct.dataclass
class OCCAMState:
    """`atari_state` is the name ppo_eval walks to reach the env state."""
    atari_state: Any
    mask_stack: jnp.ndarray


class OCCAMWrapper(JaxtariWrapper):
    """OCCAM observation wrapper, applied after AtariWrapper instead of PixelObsWrapper.

    Frame skip, max pooling, stacking, reward clipping and reward summing across
    termination match PixelObsWrapper, so runs stay comparable to the pixel baseline.
    """

    def __init__(
        self,
        env,
        mask_mode: str = "binary",
        frame_stack_size: int = 4,
        frame_skip: int = 4,
        max_pooling: bool = True,
        clip_reward: bool = True,
        out_size: Tuple[int, int] = (84, 84),
        game_name: str = "unknown",
        mods: List[str] | None = None,
        native_downscaling: bool = False,
    ):
        super().__init__(env)
        assert isinstance(env, AtariWrapper), "OCCAMWrapper must be applied after AtariWrapper"
        assert mask_mode in MASK_MODES, f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}"

        self.mask_mode = mask_mode
        self.frame_stack_size = int(frame_stack_size)
        self.frame_skip = int(frame_skip)
        self.max_pooling = bool(max_pooling)
        self.clip_reward = bool(clip_reward)
        self.out_h, self.out_w = int(out_size[0]), int(out_size[1])

        self.base_env = env._env
        # auto-reset as ObjectCentricWrapper: only run the (no-op) reset when some env is done
        self._speculative_autoreset = bool(env.noop_reset or env.first_fire)
        img_shape = self.base_env.image_space().shape
        self.img_h, self.img_w = int(img_shape[0]), int(img_shape[1])

        probe_obs, _ = self.base_env.reset(jax.random.PRNGKey(0))
        self.num_classes = len(_extract_object_groups(probe_obs))

        if self.num_classes == 0:
            raise NotImplementedError(
                f"OCCAM: game '{game_name}' exposes no ObjectObservation groups, so no "
                f"object bounding boxes are available to build masks from."
            )

        self._gray_palette = jnp.asarray(_make_gray_palette(self.num_classes))
        self._w_y = jnp.asarray(_area_weights(self.img_h, self.out_h), dtype=jnp.float32)
        self._w_x = jnp.asarray(_area_weights(self.img_w, self.out_w).T, dtype=jnp.float32)

        # opc: the gray channel from a separate copy of the game with JAXtari's native downscaled
        # grayscale renderer (as PixelObsWrapper), so the training env's renderer stays untouched
        self._gray_env = None
        if mask_mode == "opc" and native_downscaling:
            if game_name == "unknown":
                raise ValueError("OCCAM opc with native_downscaling needs game_name (and the env's mods)")
            import jaxtari
            from jaxtari.modification import apply_native_downscaling

            self._gray_env = jaxtari.make(game_name, mods=list(mods) if mods else None)
            apply_native_downscaling(self._gray_env, (self.out_h, self.out_w), True)

        self.per_frame_channels = {"planes": self.num_classes, "opc": self.num_classes + 1}.get(mask_mode, 1)
        total_channels = self.frame_stack_size * self.per_frame_channels
        self._observation_space = spaces.Box(
            low=0, high=255, shape=(total_channels, self.out_h, self.out_w, 1), dtype=jnp.uint8
        )

    def observation_space(self) -> spaces.Box:
        return self._observation_space

    def _render_small_gray(self, env_state) -> jnp.ndarray:
        """(1, out_h, out_w) uint8 frame of the native downscaled grayscale renderer (opc)."""
        return self._gray_env.render(env_state)[..., 0][None].astype(jnp.uint8)

    def _mask_single(self, frame_rgb: jnp.ndarray, obs: Any, small_gray: jnp.ndarray | None = None) -> jnp.ndarray:
        """(C, out_h, out_w) uint8 mask: drawn natively, area-downscaled, rounded like cv2.

        For opc with native downscaling, `small_gray` is the (1, out_h, out_w) gray channel.
        """
        group_masks = _group_masks(obs, self.img_h, self.img_w)

        if self.mask_mode == "object":
            gray = jnp.round(_rgb_to_gray(frame_rgb))
            native = jnp.where(_union(group_masks), gray, 0.0)[None]
        elif self.mask_mode == "binary":
            native = _union(group_masks).astype(jnp.float32)[None] * 255.0
        elif self.mask_mode == "class":
            native = self._gray_palette[_class_map(group_masks)].astype(jnp.float32)[None]
        else:  # planes, opc
            native = jnp.stack([gm.astype(jnp.float32) for gm in group_masks], axis=0) * 255.0
            if self.mask_mode == "opc" and small_gray is None:
                native = jnp.concatenate([native, jnp.round(_rgb_to_gray(frame_rgb))[None]], axis=0)

        out = _area_resize(native, self._w_y, self._w_x)
        out = jnp.clip(jnp.round(out), 0.0, 255.0).astype(jnp.uint8)
        if self.mask_mode == "opc" and small_gray is not None:
            out = jnp.concatenate([out, small_gray], axis=0)
        return out

    def _stack_to_obs(self, mask_stack: jnp.ndarray) -> jnp.ndarray:
        """(F, C, H, W) -> (F*C, H, W, 1) uint8."""
        f, c, h, w = mask_stack.shape
        return mask_stack.reshape(f * c, h, w)[..., None]

    def _reset_internal(self, key, active=True):
        # `active` = False lanes of a batched auto-reset are discarded, so they skip the no-ops
        reset_key, _ = jax.random.split(key)
        obs, atari_state = _reset_with_active(self._env, reset_key, active)
        frame = self.base_env.render(atari_state.env_state)
        small = self._render_small_gray(atari_state.env_state) if self._gray_env is not None else None
        m = self._mask_single(frame, obs, small)
        mask_stack = jnp.stack([m] * self.frame_stack_size)
        return mask_stack, OCCAMState(atari_state, mask_stack)

    @functools.partial(jax.jit, static_argnums=(0,))
    def reset(self, key) -> Tuple[jnp.ndarray, OCCAMState]:
        mask_stack, state = self._reset_internal(key)
        return self._stack_to_obs(mask_stack), state

    @functools.partial(jax.jit, static_argnums=(0,))
    def step(self, state: OCCAMState, action: int):
        # unrolled frame skip (a nested lax.scan is several times slower on GPU)
        atari_state, obs, env_states, rewards, terminations, truncations, info_dict = _unroll_frame_skip(
            self._env, state.atari_state, action, self.frame_skip
        )

        last_env_state = env_states[-1]

        # (renders a mode doesn't use are removed by XLA)
        small = None
        if self.max_pooling and self.frame_skip > 1:
            img = self.base_env.render(last_env_state)
            prev_env_state = env_states[-2]
            prev_img = self.base_env.render(prev_env_state)
            frame = jnp.maximum(img, prev_img)
            if self._gray_env is not None:
                small = jnp.maximum(self._render_small_gray(last_env_state), self._render_small_gray(prev_env_state))
        else:
            frame = self.base_env.render(last_env_state)
            if self._gray_env is not None:
                small = self._render_small_gray(last_env_state)

        new_mask = self._mask_single(frame, obs, small)
        mask_stack = jnp.concatenate([state.mask_stack[1:], new_mask[None]], axis=0)

        reward = jnp.sum(rewards)
        if self.clip_reward:
            reward = jnp.sign(reward)
        terminated = terminations.any()
        truncated = truncations.any()

        # auto-reset on game over or truncation; under vmap a plain lax.cond would run the
        # whole reset (no-ops, render, masks) for every env on every step
        done = jnp.logical_or(info_dict["env_done"], truncated)
        step_state = OCCAMState(atari_state, mask_stack)
        if self._speculative_autoreset:
            mask_stack, occ_state = speculative_cond(
                lambda done, state: done,
                lambda done, state: self._reset_internal(state.atari_state.key, done),
                lambda done, state: (state.mask_stack, state),
            )(done, step_state)
        else:
            mask_stack, occ_state = _bcast_where(
                done, self._reset_internal(atari_state.key, done), (mask_stack, step_state)
            )

        obs_out = self._stack_to_obs(mask_stack)
        return obs_out, occ_state, reward, terminated, truncated, info_dict


class _OCCAMViz:
    """[clean | obs boxes | mask (| sheets)] video frames, native or at observation resolution.

    The frames are drawn from recorded game states, which carry no observation,
    so this uses the game's private `_get_observation`; JAXtari has no public
    state -> observation call. Training and evaluation never go through it.
    """

    def __init__(self, base_env, mask_mode: str, obs_res: bool = False,
                 out_size: Tuple[int, int] = (84, 84)):
        self.env = base_env
        self.mask_mode = mask_mode
        self.obs_res = bool(obs_res)
        img_shape = base_env.image_space().shape
        self.img_h, self.img_w = int(img_shape[0]), int(img_shape[1])
        self.out_h, self.out_w = int(out_size[0]), int(out_size[1])

        probe, _ = base_env.reset(jax.random.PRNGKey(0))
        self.num_classes = len(_extract_object_groups(probe))
        self.group_names = _group_names(probe)
        self.render_order = _render_order(base_env, self.group_names)
        self._color_palette = jnp.asarray(_make_color_palette(self.num_classes))
        self._w_y = jnp.asarray(_area_weights(self.img_h, self.out_h), dtype=jnp.float32)
        self._w_x = jnp.asarray(_area_weights(self.img_w, self.out_w).T, dtype=jnp.float32)

    def _to_obs_res(self, rgb: jnp.ndarray) -> jnp.ndarray:
        """(H, W, 3) -> (out_h, out_w, 3) uint8 with the wrapper's area operator."""
        chw = jnp.transpose(rgb.astype(jnp.float32), (2, 0, 1))
        out = _area_resize(chw, self._w_y, self._w_x)
        out = jnp.clip(jnp.round(out), 0.0, 255.0).astype(jnp.uint8)
        return jnp.transpose(out, (1, 2, 0))

    def planes_rgb(self, obs: Any) -> jnp.ndarray:
        """(K, h, w) boolean planes at the active resolution."""
        masks = jnp.stack(_group_masks(obs, self.img_h, self.img_w))
        if not self.obs_res:
            return masks
        f = _area_resize(masks.astype(jnp.float32), self._w_y, self._w_x)
        return f > 0.0

    def _boxes_rgb(self, frame_rgb: jnp.ndarray, obs: Any) -> jnp.ndarray:
        """Clean frame with one class-coloured box outline per group, (H, W, 3) uint8."""
        out = frame_rgb.astype(jnp.int32)
        for gi, edge in enumerate(_group_outlines(obs, self.img_h, self.img_w)):
            out = jnp.where(edge[..., None],
                            self._color_palette[gi + 1].astype(jnp.int32), out)
        return out.astype(jnp.uint8)

    def _mask_rgb(self, frame_rgb: jnp.ndarray, obs: Any) -> jnp.ndarray:
        """Native-resolution RGB visualization (H, W, 3) uint8 of the mask."""
        group_masks = _group_masks(obs, self.img_h, self.img_w)
        if self.mask_mode == "binary":
            union = _union(group_masks)
            rgb = jnp.where(union[..., None], jnp.uint8(255), jnp.uint8(0))
            rgb = jnp.broadcast_to(rgb, (self.img_h, self.img_w, 3))

        elif self.mask_mode == "object":
            gray = _rgb_to_gray(frame_rgb)
            union = _union(group_masks)
            masked = jnp.where(union, gray, 0.0).astype(jnp.uint8)
            rgb = jnp.repeat(masked[..., None], 3, axis=-1)

        elif self.mask_mode == "class":
            rgb = self._color_palette[_class_map(group_masks)]

        else:
            rgb = self._color_palette[_class_map(group_masks, self.render_order)]

        rgb = rgb.astype(jnp.uint8)
        return self._to_obs_res(rgb) if self.obs_res else rgb

    @functools.partial(jax.jit, static_argnums=(0,))
    def _frame(self, env_state) -> jnp.ndarray:
        clean = self.env.render(env_state).astype(jnp.uint8)
        obs = self.env._get_observation(env_state)
        mask_rgb = self._mask_rgb(clean, obs)
        boxes = self._boxes_rgb(clean, obs)
        panel = self._to_obs_res(clean) if self.obs_res else clean
        boxes = self._to_obs_res(boxes) if self.obs_res else boxes
        if self.mask_mode not in ("planes", "opc"):
            return jnp.concatenate([panel, boxes, mask_rgb], axis=1)
        h, w = panel.shape[0], panel.shape[1]
        sheets = _planes_isometric_rgb(
            self.planes_rgb(obs), self._color_palette,
            h, _sheets_width(w, self.num_classes), aspect=w / h,
        )
        if self.mask_mode == "planes":
            return jnp.concatenate([panel, boxes, mask_rgb, sheets], axis=1)
        # opc: the grayscale pixel channel as a fifth panel
        gray = jnp.round(_rgb_to_gray(clean)).astype(jnp.uint8)
        gray = jnp.repeat(gray[..., None], 3, axis=-1)
        gray = self._to_obs_res(gray) if self.obs_res else gray
        return jnp.concatenate([panel, boxes, mask_rgb, sheets, gray], axis=1)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _counts(self, env_state) -> jnp.ndarray:
        """(K,) int32 active, on-screen boxes per group."""
        obs = self.env._get_observation(env_state)
        return jnp.stack([
            _group_box_arrays(g, self.img_h, self.img_w)[4].sum().astype(jnp.int32)
            for g in _extract_object_groups(obs)
        ])

    def counts(self, env_states) -> jnp.ndarray:
        """(T, ...) base env states -> (T, K) int32."""
        return jax.vmap(self._counts)(env_states)

    def panel_layout(self) -> List[Tuple[str, int]]:
        """[(panel name, width)] in the order _frame concatenates them."""
        w = self.out_w if self.obs_res else self.img_w
        base = [("clean", w), ("obs boxes", w), (self.mask_mode, w)]
        if self.mask_mode not in ("planes", "opc"):
            return base
        sheets = [("planes (sheets)", _sheets_width(w, self.num_classes))]
        return base + sheets + ([("gray", w)] if self.mask_mode == "opc" else [])

    def frames(self, env_states) -> jnp.ndarray:
        """(T, ...) base env states -> (T, H, W_row, 3) uint8; 4 panels in planes mode, 5 in opc."""
        return jax.vmap(self._frame)(env_states)


def occam_comparison_frames(env_id: str, mask_mode: str, env_states, mods=None):
    """Eval rollout -> (T, H, W_row, 3) uint8 frames with a colour key and panel headers."""
    import jaxtari

    base_env = jaxtari.make(env_id, mods=mods or None)
    viz = _OCCAMViz(base_env, mask_mode)
    frames = np.asarray(viz.frames(env_states), dtype=np.uint8)
    counts = np.asarray(viz.counts(env_states), dtype=np.int32)
    palette = np.asarray(_make_color_palette(viz.num_classes))
    layout = viz.panel_layout()
    total_w = frames.shape[2]

    header = np.concatenate([_strip(w, HEAD_H, name) for name, w in layout], axis=1)
    header = np.pad(header, [(0, 0), (0, max(0, total_w - header.shape[1])), (0, 0)])[:, :total_w]

    cache = {}
    out = []
    for t in range(frames.shape[0]):
        key = tuple(int(c) for c in counts[t])
        if key not in cache:
            entries = [(n, key[i], tuple(int(c) for c in palette[i + 1]))
                       for i, n in enumerate(viz.group_names)]
            cache[key] = _legend_strip(total_w, entries)
        out.append(np.concatenate([cache[key], header, frames[t]], axis=0))
    return np.stack(out)


def _to_chw(frames_thwc: np.ndarray) -> np.ndarray:
    """(T, H, W, 3) -> (T, 3, H, W) contiguous, for wandb.Video."""
    return np.ascontiguousarray(np.transpose(frames_thwc, (0, 3, 1, 2)))


LEG_LINE_H = 15
HEAD_H = 16


def _legend_height(n_entries: int) -> int:
    """Height of _legend_strip for n entries."""
    n = max(1, int(n_entries))
    rows = 1 if n <= 6 else (n + 5) // 6
    return rows * LEG_LINE_H + 4


def _strip(width, height, text, fill=(255, 255, 255), font_size=13):
    """Black banner of the given size with left-aligned text."""
    band = np.zeros((height, width, 3), np.uint8)
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(band)
        d = ImageDraw.Draw(im)
        d.text((4, max(0, (height - font_size) // 2 - 1)), text,
               fill=fill, font=_load_font(font_size))
        band = np.asarray(im)
    except Exception:
        pass
    return band


def _legend_strip(width, entries, font_size=13):
    """Colour key for the object groups: [(name, n_active, rgb)] -> black banner."""
    n = max(1, len(entries))
    rows = 1 if n <= 6 else (n + 5) // 6
    band = np.zeros((_legend_height(n), width, 3), np.uint8)
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return band
    im = Image.fromarray(band)
    d = ImageDraw.Draw(im)
    font = _load_font(font_size)
    prefix = "obs boxes:"
    d.text((5, 3), prefix, fill=(150, 150, 150), font=font)
    x = 5 + int(d.textlength(prefix, font=font)) + 10
    y = 2
    per_row = (n + rows - 1) // rows
    for i, (name, n_act, col) in enumerate(entries):
        if i and i % per_row == 0:
            x, y = 5, y + LEG_LINE_H
        d.rectangle([x, y + 4, x + 7, y + 11], fill=col)
        text = name if n_act is None else f"{name} ({n_act})"
        d.text((x + 13, y + 1), text, fill=col, font=font)
        x += 13 + int(d.textlength(text, font=font)) + 16
    return np.asarray(im)


def _load_font(size: int):
    """TrueType font with fallback to PIL bitmap default."""
    try:
        from PIL import ImageFont
        for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except Exception:
                continue
        return ImageFont.load_default()
    except Exception:
        return None


def _caption_clip(frames_thwc: np.ndarray, text: str, banner_h: int = 16) -> np.ndarray:
    """Prepend a caption banner to every frame and pad H, W to even sizes."""
    T, H, W, C = frames_thwc.shape
    banner = np.zeros((banner_h, W, 3), np.uint8)
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(banner)
        ImageDraw.Draw(im).text((3, 2), text, fill=(255, 255, 255))
        banner = np.asarray(im)
    except Exception:
        pass
    banner = np.broadcast_to(banner, (T, banner_h, W, 3))
    out = np.concatenate([banner, frames_thwc], axis=1)
    return np.pad(out, [(0, 0), (0, out.shape[1] % 2), (0, out.shape[2] % 2), (0, 0)])


def log_occam_comparison_video(
    env_id: str,
    mask_mode: str,
    env_states,
    mods=None,
    mod_label: str = "default",
    step: int = 0,
    fps: int = 30,
):
    """Render, caption and log a [clean | boxes | mask] eval clip to W&B as eval/video_occam_<mod_label>."""
    import wandb

    frames = occam_comparison_frames(env_id, mask_mode, env_states, mods=mods)
    captioned = _caption_clip(frames, f"{env_id} | {mask_mode} | {mod_label} | step {step}")
    wandb.log({f"eval/video_occam_{mod_label}": wandb.Video(_to_chw(captioned), fps=fps, format="mp4")}, step=step)
    print(f"OCCAM comparison video logged to wandb with {frames.shape[0]} frames ({mod_label}).")
