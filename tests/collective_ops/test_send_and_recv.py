import pytest

from mpi4py import MPI

import jax
import jax.numpy as jnp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv():
    from mpi4jax import recv, send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    if rank == 0:
        for proc in range(1, size):
            res, token = recv(arr, source=proc, tag=proc)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send(arr, 0, tag=rank)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_scalar():
    from mpi4jax import recv, send

    arr = 1 * rank
    _arr = 1 * rank

    if rank == 0:
        for proc in range(1, size):
            res, token = recv(arr, source=proc, tag=proc)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send(arr, 0, tag=rank)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_scalar_jit():
    from mpi4jax import recv, send

    arr = 1 * rank
    _arr = 1 * rank

    @jax.jit
    def send_jit(x):
        send(x, 0, tag=rank)
        return x

    if rank == 0:
        for proc in range(1, size):
            res = jax.jit(lambda x: recv(x, source=proc, tag=proc)[0])(arr)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send_jit(arr)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_jit():
    from mpi4jax import recv, send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    @jax.jit
    def send_jit(x):
        send(x, 0, tag=rank)
        return x

    if rank == 0:
        for proc in range(1, size):
            res = jax.jit(lambda x: recv(x, source=proc, tag=proc)[0])(arr)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send_jit(arr)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_send_recv_deadlock():
    from mpi4jax import recv, send

    # this deadlocks without proper token management
    @jax.jit
    def deadlock(arr):
        if rank == 0:
            # send, then receive
            _, token = send(arr, 1)
            newarr, _ = recv(arr, 1, token=token)
        else:
            # receive, then send
            newarr, token = recv(arr, 0)
            send(arr, 0, token=token)
        return newarr

    arr = jnp.ones(10) * rank
    arr = deadlock(arr)
    assert jnp.array_equal(arr, jnp.ones_like(arr) * (1 - rank))


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_status():
    from mpi4jax import recv, send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    if rank == 0:
        for proc in range(1, size):
            status = MPI.Status()
            res, token = recv(arr, source=proc, tag=proc, status=status)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
            assert status.Get_source() == proc
    else:
        send(arr, 0, tag=rank)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_status_jit():
    from mpi4jax import recv, send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()

    @jax.jit
    def send_jit(x):
        send(x, 0, tag=rank)
        return x

    if rank == 0:
        for proc in range(1, size):
            status = MPI.Status()
            res = jax.jit(lambda x: recv(x, source=proc, tag=proc, status=status)[0])(
                arr
            )
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
            assert status.Get_source() == proc
    else:
        send_jit(arr)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_send_recv_jvp():
    from mpi4jax import recv, send

    arr = rank * jnp.ones((3,))

    darr = jnp.zeros((3,))
    if rank == 0:
        darr = darr.at[2].set(1)

    def exchange(x):
        if rank == 0:
            x, token = send(x, 1)
            x_new, _ = recv(x, 1, token=token)
        else:
            x_new, token = recv(x, 0)
            send(x, 0, token=token)
        return x_new
    
    primals, tangents = jax.jvp(exchange, (arr,), (darr,))

    if rank == 0:
        assert jnp.array_equal(primals, jnp.ones((3,)))
        assert jnp.array_equal(tangents, jnp.zeros((3,)))
    else:
        assert jnp.array_equal(primals, jnp.zeros((3,)))
        hypothesis = jnp.zeros((3,))
        hypothesis = hypothesis.at[2].set(1)
        assert jnp.array_equal(tangents, hypothesis)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_send_recv_vjp():
    from mpi4jax import recv, send

    arr = rank * jnp.ones((3,))

    Darr = jnp.zeros((3,))
    if rank == 0:
        Darr = Darr.at[2].set(1)

    def exchange(x):
        if rank == 0:
            x, token= send(x, 1, tag = 1)
            x_new, _ = recv(x, 1, tag = 2, token=token)
        else:
            x_new, token = recv(x, 0, tag = 1)
            x, _ = send(x, 0, tag = 2, token=token)
        return x_new, x
    
    primals, exchange_vjp = jax.vjp(exchange, arr)
    null_arg = jnp.zeros((3,))
    cotangents = exchange_vjp((Darr, null_arg))

    if rank == 0:
        assert jnp.array_equal(primals[0], jnp.ones((3,)))
        assert jnp.array_equal(primals[1], arr)
        assert jnp.array_equal(cotangents[0], jnp.zeros((3,)))
    else:
        assert jnp.array_equal(primals[0], jnp.zeros((3,)))
        assert jnp.array_equal(primals[1], arr)
        hypothesis = jnp.zeros((3,))
        hypothesis = hypothesis.at[2].set(1)
        assert jnp.array_equal(cotangents[0], hypothesis)