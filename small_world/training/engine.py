import os
import shutil
import time
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import random
from jax.lax import scan

# Generic training engine: knows nothing about equations or architectures.
#
# The problem is described by
#   arrays:   pytree of device arrays (data), passed to jitted functions as arguments, not closed
#             over, so that they are not baked into the compiled program;
#   loss_fn:  (model, arrays, indices) -> scalar, loss on the samples `indices` (batch);
#   error_fn: (model, arrays, indices) -> (len(indices),) evaluation error per sample.
# Training runs in chunks of `stop_each` epochs, each chunk is one jitted `scan` over the
# optimisation steps. After a chunk the model is evaluated on the validation samples and a
# checkpoint is written to disk (never kept in memory), so the best model can be restored.


def make_optimizer(name, learning_rate, decay_steps, gamma):
    schedule = optax.exponential_decay(learning_rate, decay_steps, gamma)
    if name == "lion":
        return optax.lion(learning_rate=schedule)
    if name == "adam":
        return optax.adam(learning_rate=schedule)
    raise ValueError(f"unknown optimiser {name!r}, expected 'adam' or 'lion'")


def make_batches(key, N_train, N_batch, N_epoch):
    """Indices (N_epoch * (N_train // N_batch), N_batch): every epoch is a fresh permutation of the train set."""
    steps = N_train // N_batch
    perms = jax.vmap(lambda k: random.permutation(k, N_train))(random.split(key, N_epoch))
    return perms[:, :steps * N_batch].reshape(N_epoch * steps, N_batch)


def count_parameters(model):
    leaves = jax.tree.leaves(eqx.filter(model, eqx.is_array))
    return sum(2 * leaf.size if jnp.iscomplexobj(leaf) else leaf.size for leaf in leaves)


@eqx.filter_jit
def _run_chunk(model, opt_state, arrays, batches, loss_fn, optim):
    params, static = eqx.partition(model, eqx.is_array)

    def step(carry, indices):
        params, opt_state = carry
        loss, grads = eqx.filter_value_and_grad(lambda p: loss_fn(eqx.combine(p, static), arrays, indices))(params)
        grads = jax.tree.map(lambda g: g.conj(), grads)
        updates, opt_state = optim.update(grads, opt_state, params)
        return (eqx.apply_updates(params, updates), opt_state), loss

    (params, opt_state), history = scan(step, (params, opt_state), batches)
    return eqx.combine(params, static), opt_state, history


def save_checkpoint(path, model, opt_state):
    os.makedirs(path, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(path, "model.eqx"), model)
    eqx.tree_serialise_leaves(os.path.join(path, "opt_state.eqx"), opt_state)


def load_checkpoint(path, model_like, optim):
    """Model and optimiser state saved by save_checkpoint; model_like only provides the structure."""
    model = eqx.tree_deserialise_leaves(os.path.join(path, "model.eqx"), model_like)
    opt_state_like = optim.init(eqx.filter(model_like, eqx.is_array))
    opt_state = eqx.tree_deserialise_leaves(os.path.join(path, "opt_state.eqx"), opt_state_like)
    return model, opt_state


class FitResult(NamedTuple):
    model: eqx.Module       # best model by validation error (restored from disk)
    opt_state: object
    best_n: int             # index of the best evaluation (chunk)
    val_errors: np.ndarray  # mean validation error after every chunk (nan if diverged)
    history: np.ndarray     # training loss at every step
    training_time: float    # seconds spent in training (includes compilation, excludes evaluation)
    stop_reason: str        # 'finished', 'patience' or 'nan'
    best_path: str          # directory of the best checkpoint


def fit(model, optim, loss_fn, error_fn, arrays, N_train, val_indices, *, N_batch, N_epoch, stop_each,
        key, ckpt_dir, patience=None, keep_checkpoints=False, log=print):
    """Train on samples [0, N_train) of `arrays`.

    Every `stop_each` epochs: evaluate on `val_indices`, write a checkpoint ckpt_dir/chunk_XXXX.
    Stops early if the validation error is nan or has not improved for `patience` evaluations
    (None disables). At the end the best checkpoint is copied to ckpt_dir/best and the model is
    restored from it; the chunk checkpoints are removed unless keep_checkpoints."""
    evaluate = eqx.filter_jit(error_fn)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    n_chunks = -(-N_epoch // stop_each)
    keys = random.split(key, n_chunks)
    val_errors, histories, chunk_paths = [], [], []
    best, best_n, since_best, training_time, stop_reason = np.inf, 0, 0, 0.0, "finished"

    for n in range(n_chunks):
        epochs = min(stop_each, N_epoch - n * stop_each)
        batches = make_batches(keys[n], N_train, N_batch, epochs)
        start = time.time()
        model, opt_state, history = jax.block_until_ready(_run_chunk(model, opt_state, arrays, batches, loss_fn, optim))
        training_time += time.time() - start
        histories.append(np.asarray(history))

        val_error = float(jnp.mean(evaluate(model, arrays, val_indices)))
        val_errors.append(val_error)
        chunk_paths.append(os.path.join(ckpt_dir, f"chunk_{n:04d}"))
        save_checkpoint(chunk_paths[-1], model, opt_state)
        log(f"epoch {(n + 1) * stop_each:>6}  train loss {histories[-1].mean():.4e}  val error {val_error:.4e}  time {training_time:.1f}s")

        if np.isnan(val_error):
            stop_reason = "nan"
            break
        if val_error < best:
            best, best_n, since_best = val_error, n, 0
        else:
            since_best += 1
        if patience is not None and since_best >= patience:
            stop_reason = "patience"
            break

    best_path = os.path.join(ckpt_dir, "best")
    if os.path.exists(best_path):
        shutil.rmtree(best_path)
    shutil.copytree(chunk_paths[best_n], best_path)
    if not keep_checkpoints:
        for path in chunk_paths:
            shutil.rmtree(path)
    model, opt_state = load_checkpoint(best_path, model, optim)
    return FitResult(model, opt_state, best_n, np.array(val_errors), np.concatenate(histories),
                     training_time, stop_reason, best_path)


if __name__ == "__main__":
    import tempfile

    # toy problem: fit y = A x with a linear model, checks of checkpointing and stopping
    rng = np.random.default_rng(0)
    X = jnp.asarray(rng.normal(size=(200, 5)), dtype=jnp.float32)
    A = jnp.asarray(rng.normal(size=(3, 5)), dtype=jnp.float32)
    arrays = {"x": X, "y": X @ A.T}
    loss_fn = lambda model, arrays, ind: jnp.mean((jax.vmap(model)(arrays["x"][ind]) - arrays["y"][ind])**2)
    error_fn = lambda model, arrays, ind: jnp.linalg.norm(jax.vmap(model)(arrays["x"][ind]) - arrays["y"][ind], axis=1) / jnp.linalg.norm(arrays["y"][ind], axis=1)
    make_model = lambda: eqx.nn.Linear(5, 3, key=random.PRNGKey(1))
    val = jnp.arange(150, 200)
    kwargs = dict(N_batch=10, N_epoch=40, stop_each=5, key=random.PRNGKey(2), log=lambda *a: None)

    with tempfile.TemporaryDirectory() as tmp:
        optim = make_optimizer("adam", 1e-2, 1000, 0.5)
        res = fit(make_model(), optim, loss_fn, error_fn, arrays, 150, val, ckpt_dir=f"{tmp}/a", keep_checkpoints=True, **kwargs)
        chunks = sorted(os.listdir(f"{tmp}/a"))
        print("trained, val errors", res.val_errors[[0, -1]], "history", res.history.shape, "stop", res.stop_reason)
        assert res.val_errors[-1] < res.val_errors[0] and res.stop_reason == "finished"
        assert chunks == ["best"] + [f"chunk_{i:04d}" for i in range(8)], chunks
        assert res.history.shape == (40 * (150 // 10),)

        # the restored best model reproduces the recorded validation error
        restored, _ = load_checkpoint(res.best_path, make_model(), optim)
        assert np.isclose(float(error_fn(restored, arrays, val).mean()), res.val_errors[res.best_n], atol=1e-4)

        # only the best checkpoint is kept
        res = fit(make_model(), optim, loss_fn, error_fn, arrays, 150, val, ckpt_dir=f"{tmp}/b", keep_checkpoints=False, **kwargs)
        assert os.listdir(f"{tmp}/b") == ["best"]

        # early stopping: a huge learning rate makes the validation error stall
        optim_bad = make_optimizer("adam", 1e2, 1000, 0.5)
        res = fit(make_model(), optim_bad, loss_fn, error_fn, arrays, 150, val, ckpt_dir=f"{tmp}/c", patience=2, **kwargs)
        print("patience: stopped after", len(res.val_errors), "evaluations, reason", res.stop_reason)
        assert res.stop_reason == "patience" and len(res.val_errors) < 8

        # divergence is detected
        nan_loss = lambda model, arrays, ind: loss_fn(model, arrays, ind) * jnp.nan
        res = fit(make_model(), optim, nan_loss, error_fn, arrays, 150, val, ckpt_dir=f"{tmp}/d", **kwargs)
        assert res.stop_reason == "nan" and len(res.val_errors) == 1 and res.best_n == 0
    print("all checks passed")
