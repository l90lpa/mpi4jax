import pytest

from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()
    token = jnp.empty((1,))

    other = 1 - rank

    res, token = sendrecv(arr, arr, token, source=other, dest=other)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_status():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()
    token = jnp.empty((1,))

    other = 1 - rank

    status = MPI.Status()
    res, token = sendrecv(arr, arr, token, source=other, dest=other, status=status)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)
    assert status.Get_source() == other


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_status_jit():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()
    token = jnp.empty((1,))

    other = 1 - rank

    status = MPI.Status()
    res = jax.jit(
        lambda x, y: sendrecv(x, y, token, source=other, dest=other, status=status)[0]
    )(arr, arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)
    assert status.Get_source() == other


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_scalar():
    from mpi4jax import sendrecv

    arr = 1 * rank
    _arr = arr
    token = jnp.empty((1,))

    other = 1 - rank

    res, token = sendrecv(arr, arr, token, source=other, dest=other)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_jit():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()
    token = jnp.empty((1,))

    other = 1 - rank

    res = jax.jit(lambda x, y: sendrecv(x, y, token, source=other, dest=other)[0])(arr, arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_scalar_jit():
    from mpi4jax import sendrecv

    arr = 1 * rank
    _arr = arr
    token = jnp.empty((1,))

    other = 1 - rank

    res = jax.jit(lambda x, y: sendrecv(x, y, token, source=other, dest=other)[0])(arr, arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_vmap():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()
    token = jnp.empty((1,))

    other = 1 - rank

    res = sendrecv(arr, arr, token, source=other, dest=other)[0]

    def fun(x, y):
        return sendrecv(x, y, token, source=other, dest=other)[0]

    vfun = jax.vmap(fun, in_axes=(0, 0))
    res = vfun(_arr, arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * other)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_grad():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * (rank + 1)
    _arr = arr.copy()

    other = 1 - rank

    def f(x):
        x, _ = sendrecv(x, x, jnp.empty((1,)), source=other, dest=other)
        x = x * (rank + 1)
        return x.sum()

    res = jax.grad(f)(arr)

    assert jnp.array_equal(res, jnp.ones_like(arr) * (other + 1))
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_grad_2():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * (rank + 1)
    _arr = arr.copy()
    token = jnp.empty((1,))

    other = 1 - rank

    def f(x, tok):
        x, tok = sendrecv(x, x, tok, source=other, dest=other)
        x = x * (rank + 1) * 5
        x, tok = sendrecv(x, x, tok, source=other, dest=other)
        x = x * (rank + 1) ** 2
        return x.sum()

    res = jax.grad(f)(arr, token)

    solution = (rank + 1) ** 2 * (other + 1) * 5
    print("solution is ", solution)
    assert jnp.array_equal(res, jnp.ones_like(arr) * solution)
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_jacfwd():
    from mpi4jax import sendrecv

    arr = jnp.ones((2,)) * (rank + 1)
    _arr = arr.copy()
    token = jnp.empty((1,))

    other = 1 - rank

    def f(x, tok):
        x, tok = sendrecv(x, x, tok, source=other, dest=other)
        return x, tok

    res = jax.jacfwd(f)(arr, token)[0]

    jnp.array_equal(res, jnp.identity(2))
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_sendrecv_jacrev():
    from mpi4jax import sendrecv

    arr = jnp.ones((3, 2)) * (rank + 1)
    _arr = arr.copy()
    token = jnp.empty((1,))

    other = 1 - rank

    def f(x, tok):
        x, tok = sendrecv(x, x, tok, source=other, dest=other)
        x = x * (rank + 1)
        return x.sum(), tok

    res = jax.jacrev(f)(arr, token)[0]

    assert jnp.array_equal(res, jnp.ones_like(arr) * (other + 1))
    assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 3, reason="To ensure the source and destination combination are distinct on each process")
def test_sendrecv_jvp():
    from mpi4jax import sendrecv

    x = jnp.ones((2,)) * (rank + 1)
    _x = x.copy()
    token = jnp.empty((1,))

    dx = jnp.ones((2,)) * -1 * (rank + 1)
    dtoken = jnp.empty((1,))

    left = (rank - 1) % size
    right = (rank + 1) % size

    def f(x, tok):
        x, tok = sendrecv(x, x, tok, source=left, dest=right)
        return x, tok


    y, token, dy, dtoken = jax.jvp(f, (x, token), (dx, dtoken))

    assert jnp.array_equal(dy, jnp.ones_like(x) * -1 * (left + 1))
    assert jnp.array_equal(_x, x)


@pytest.mark.skipif(size < 3, reason="To ensure the source and destination combination are distinct on each process")
def test_sendrecv_vjp():
    from mpi4jax import sendrecv

    x = jnp.ones((3,2)) * (rank + 1)
    _x = x.copy()
    token = jnp.empty((1,))

    Dy = jnp.ones((3,2)) * -1 * (rank + 1)
    Dtoken = jnp.empty((1,))

    left = (rank - 1) % size
    right = (rank + 1) % size

    def f(x, tok):
        x, tok = sendrecv(x, x, tok, source=left, dest=right)
        return x, tok


    _, vjp_f = jax.vjp(f, x, token)
    Dx = vjp_f((Dy, Dtoken))[0]

    assert jnp.array_equal(Dx, jnp.ones_like(x) * -1 * (right + 1))
    assert jnp.array_equal(_x, x)