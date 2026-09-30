import jax
import jax.numpy as jnp
import chex
import optax

@chex.dataclass(frozen=True)
class LamdbReturn:
    value : chex.Array
    reward : chex.Array
    cont : chex.Array


def symlog(x : jnp.ndarray, alpha : float = 1.):
    return jnp.sign(x) * jnp.log1p(jnp.abs(alpha * x))

def symexp(x : jnp.ndarray, alpha : float = 1.):
    return jnp.sign(x) * (jnp.expm1(jnp.abs(alpha * x)))

def lambda_return(
        rewards,
        dones,
        values,
        gamma : float = 0.99,
        lambd : float = 0.95
    ):

    init_value = jax.lax.cond(
        dones[-1] == 1,
        lambda: 0.,
        lambda: values[-1]
    )

    _init = LamdbReturn(
        reward=rewards, #r_t (0..T-1)
        value=values[:-1], # v_t+1 (1.. T)
        cont=(1-dones) # c_t (0..T-1)
    )
    def _scan_fn(returns, lambd_return):
        returns = lambd_return.reward + \
            gamma * lambd_return.cont * ((1.-lambd) * lambd_return.value + lambd * returns)
        return returns, returns

    lambda_returns, returns = jax.lax.scan(
        _scan_fn,
        init_value,
        _init,
        reverse=True
    )

    return jnp.concatenate([returns, init_value[None]], axis=-1)


def twohot(x, bins):
    x = jnp.clip(x, bins[0], bins[-1])
    k = jnp.sum(bins < x) - 1

    def _twohot(k):
        delta = jnp.abs(bins[k+1] - bins[k])
        encoding = jnp.zeros_like(bins, dtype=jnp.float32)
        encoding = encoding.at[k].add((bins[k+1]-x)/ delta)
        encoding = encoding.at[k+1].add((x-bins[k])/ delta)
        return encoding
    
    encoding = jax.lax.cond(
        k >= 0,
        _twohot,
        lambda k : jnp.zeros_like(bins).at[0].add(1.),
        k
    )
    return encoding


def scale_by_momentum(beta=0.9, nesterov=False):

    def init_fn(params):
        mu = jax.tree.map(lambda t: jnp.zeros_like(t, jnp.float32), params)
        step = jnp.zeros((), jnp.int32)
        return (step, mu)

    def update_fn(updates, state, params=None):
        step, mu = state
        step = optax.safe_int32_increment(step)
        mu = optax.update_moment(updates, mu, beta, 1)
        if nesterov:
            mu_nesterov = optax.update_moment(updates, mu, beta, 1)
            mu_hat = optax.bias_correction(mu_nesterov, beta, step)
        else:
            mu_hat = optax.bias_correction(mu, beta, step)
        return mu_hat, (step, mu)

    return optax.GradientTransformation(init_fn, update_fn)


def inv_twohot(probs, bins, transform=lambda x: x):
    n = probs.shape[-1]
    if n % 2 == 1:
        m = (n - 1) // 2
        p1 = probs[..., :m]
        p2 = probs[..., m: m + 1]
        p3 = probs[..., m + 1:]
        b1 = bins[..., :m]
        b2 = bins[..., m: m + 1]
        b3 = bins[..., m + 1:]
        wavg = (p2 * b2).sum(-1) + ((p1 * b1)[..., ::-1] + (p3 * b3)).sum(-1)
        return transform(wavg)
    else:
        p1 = probs[..., :n // 2]
        p2 = probs[..., n // 2:]
        b1 = bins[..., :n // 2]
        b2 = bins[..., n // 2:]
        wavg = ((p1 * b1)[..., ::-1] + (p2 * b2)).sum(-1)
        return transform(wavg)

