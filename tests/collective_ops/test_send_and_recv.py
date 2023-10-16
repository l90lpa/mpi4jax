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
    token = jnp.empty((1,))

    if rank == 0:
        for proc in range(1, size):
            res, token = recv(arr, token, source=proc, tag=proc)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send(arr, token, 0, tag=rank)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_scalar():
    from mpi4jax import recv, send

    arr = 1 * rank
    _arr = 1 * rank
    token = jnp.empty((1,))

    if rank == 0:
        for proc in range(1, size):
            res, token = recv(arr, token, source=proc, tag=proc)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send(arr, token, 0, tag=rank)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_scalar_jit():
    from mpi4jax import recv, send

    arr = 1 * rank
    _arr = 1 * rank
    token = jnp.empty((1,))

    @jax.jit
    def send_jit(x, tok):
        tok = send(x, tok, 0, tag=rank)
        return x, tok

    if rank == 0:
        for proc in range(1, size):
            def recv_jit(x, tok):
                x, tok = recv(x, tok, source=proc, tag=proc)
                return x, tok
            res = recv_jit(arr, token)
            assert jnp.array_equal(res[0], jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send_jit(arr, token)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_jit():
    from mpi4jax import recv, send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()
    token = jnp.empty((1,))

    @jax.jit
    def send_jit(x, tok):
        tok = send(x, tok, 0, tag=rank)
        return x, tok

    if rank == 0:
        for proc in range(1, size):
            @jax.jit
            def recv_jit(x, tok):
                x, tok = recv(x, tok, source=proc, tag=proc)
                return x, tok
            res = recv_jit(arr, token)
            assert jnp.array_equal(res[0], jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
    else:
        send_jit(arr, token)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_send_recv_deadlock():
    from mpi4jax import recv, send

    # this deadlocks without proper token management
    @jax.jit
    def deadlock(arr, tok):
        if rank == 0:
            # send, then receive
            tok = send(arr, tok, 1)
            newarr, tok = recv(arr, tok, 1)
        else:
            # receive, then send
            newarr, tok = recv(arr, tok, 0)
            tok = send(arr, tok, 0)
        return newarr, tok

    arr = jnp.ones(10) * rank
    token = jnp.empty((1,))
    arr, token = deadlock(arr, token)
    assert jnp.array_equal(arr, jnp.ones_like(arr) * (1 - rank))


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_status():
    from mpi4jax import recv, send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()
    token = jnp.empty((1,))

    if rank == 0:
        for proc in range(1, size):
            status = MPI.Status()
            res, token = recv(arr, token, source=proc, tag=proc, status=status)
            assert jnp.array_equal(res, jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
            assert status.Get_source() == proc
    else:
        send(arr, token, 0, tag=rank)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2, reason="need at least 2 processes to test send/recv")
def test_send_recv_status_jit():
    from mpi4jax import recv, send

    arr = jnp.ones((3, 2)) * rank
    _arr = arr.copy()
    token = jnp.empty((1,))

    @jax.jit
    def send_jit(x, tok):
        tok = send(x, tok, 0, tag=rank)
        return x, tok

    if rank == 0:
        for proc in range(1, size):
            status = MPI.Status()
            @jax.jit
            def recv_jit(x, tok):
                x, tok = recv(x, tok, source=proc, tag=proc, status=status)
                return x, tok
            res = recv_jit(arr, token)
            assert jnp.array_equal(res[0], jnp.ones_like(arr) * proc)
            assert jnp.array_equal(_arr, arr)
            assert status.Get_source() == proc
    else:
        send_jit(arr, token)
        assert jnp.array_equal(_arr, arr)


@pytest.mark.skipif(size < 2 or rank > 1, reason="Runs only on rank 0 and 1")
def test_send_recv_jvp():
    from mpi4jax import recv, send

    arr = rank * jnp.ones((3,))
    token = jnp.empty((1,))

    darr = jnp.zeros((3,))
    if rank == 0:
        darr = darr.at[2].set(1)
    dtoken = jnp.empty((1,))

    def exchange(x, tok):
        if rank == 0:
            tok = send(x, tok, 1)
            x_new, tok = recv(x, tok, 1)
        else:
            x_new, tok = recv(x, tok, 0)
            tok = send(x, tok, 0)
        return x_new, tok
    
    primals, tangents = jax.jvp(exchange, (arr,token), (darr,dtoken))
    primals = primals[0]
    tangents = tangents[0]

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
    token = jnp.empty((1,))

    Darr = jnp.zeros((3,))
    if rank == 0:
        Darr = Darr.at[2].set(1)
    Dtoken = jnp.empty((1,))

    def exchange(x, tok):
        if rank == 0:
            tok        = send(x, tok, 1, tag = 1) # x_t        , tok_t += recv(zero   , tok_t)
            x_new, tok = recv(x, tok, 1, tag = 2) # x_t += zero, tok_t += send(x_new_t, tok_t)
        else:
            x_new, tok = recv(x, tok, 0, tag = 1)
            tok        = send(x, tok, 0, tag = 2)
        return x_new, tok
    
    primals, exchange_vjp = jax.vjp(exchange, arr, token)
    cotangents = exchange_vjp((Darr, Dtoken))

    if rank == 0:
        assert jnp.array_equal(primals[0], jnp.ones((3,)))
        assert jnp.array_equal(cotangents[0], jnp.zeros((3,)))
    else:
        assert jnp.array_equal(primals[0], jnp.zeros((3,)))
        hypothesis = jnp.zeros((3,))
        hypothesis = hypothesis.at[2].set(1)
        assert jnp.array_equal(cotangents[0], hypothesis)