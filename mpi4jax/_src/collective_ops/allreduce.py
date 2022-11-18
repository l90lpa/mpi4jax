import numpy as _np
from mpi4py import MPI as _MPI

from jax import abstract_arrays, core
from jax.core import Primitive, Tracer, Token
from jax.interpreters import ad, xla, batching
from jax.lax import create_token
from jax.lib import xla_client

from ..utils import (
    HashableMPIType,
    default_primitive_impl,
    to_dtype_handle,
    to_mpi_handle,
    unpack_hashable,
    wrap_as_hashable,
    xla_constant_intc,
    xla_constant_uintptr,
)
from ..decorators import translation_rule_cpu, translation_rule_gpu, mpi4jax_debug
from ..validation import enforce_types
from ..comm import get_default_comm
from ..jax_compat import register_abstract_eval

# The Jax primitive
mpi_allreduce_p = Primitive("allreduce_mpi")  # Create the primitive
mpi_allreduce_impl = default_primitive_impl(mpi_allreduce_p)


# This function applies the primitive to an AST
@enforce_types(
    op=(_MPI.Op, HashableMPIType),
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    token=(type(None), Token, Tracer),
)
def allreduce(x, op, *, comm=None, token=None):
    """Perform an allreduce operation.

    .. note::

       This primitive can be differentiated via :func:`jax.grad` and related functions
       if ``op`` is :obj:`mpi4py.MPI.SUM`.

    Arguments:
        x: Array or scalar input.
        op (mpi4py.MPI.Op): The reduction operator (e.g :obj:`mpi4py.MPI.SUM`).
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        token (Token): XLA token to use to ensure correct execution order.
            If not given, a new token is generated.

    Returns:
        Tuple[DeviceArray, Token]:
            - Result of the allreduce operation.
            - A new, modified token, that depends on this operation.

    """
    if token is None:
        token = create_token(x)

    if comm is None:
        comm = get_default_comm()

    if mpi4jax_debug:
        print(f"r{comm.Get_rank()}| Ar->scheduled for {x.shape}:{x.dtype}")

    op = wrap_as_hashable(op)
    comm = wrap_as_hashable(comm)
    return tuple(mpi_allreduce_p.bind(x, token, op=op, comm=comm, transpose=False))


# This function compiles the operation
# transpose is a boolean flag that signals whever this is the forward pass
# performing the MPI reduction, or the transposed pass, which is trivial
@translation_rule_cpu
def mpi_allreduce_xla_encode_cpu(c, x, token, op, comm, transpose):
    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    if transpose:
        assert op == _MPI.SUM
        return xla_client.ops.Tuple(c, [x, token])

    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)

    sh = xla_client.Shape.tuple_shape(
        [xla_client.Shape.array_shape(dtype, dims), xla_client.Shape.token_shape()]
    )

    if mpi4jax_debug:
        print(f"r{comm.Get_rank()}| Ar->encoded_cpu")

    return xla_client.ops.CustomCall(
        c,
        b"mpi_allreduce",
        operands=(
            xla_constant_intc(c, nitems),
            x,
            xla_constant_uintptr(c, to_mpi_handle(op)),
            xla_constant_uintptr(c, to_mpi_handle(comm)),
            xla_constant_uintptr(c, to_dtype_handle(dtype)),
            token,
        ),
        shape=sh,
        has_side_effect=True,
    )


@translation_rule_gpu
def mpi_allreduce_xla_encode_gpu(c, x, token, op, comm, transpose):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_allreduce_descriptor

    op = unpack_hashable(op)
    comm = unpack_hashable(comm)

    if transpose:
        assert op == _MPI.SUM
        return xla_client.ops.Tuple(c, [x, token])

    x_shape = c.GetShape(x)
    dtype = x_shape.element_type()
    dims = x_shape.dimensions()

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)

    sh = xla_client.Shape.tuple_shape(
        [xla_client.Shape.array_shape(dtype, dims), xla_client.Shape.token_shape()]
    )

    descriptor = build_allreduce_descriptor(
        _np.intc(nitems),
        to_mpi_handle(op),
        to_mpi_handle(comm),
        to_dtype_handle(dtype),
    )

    if mpi4jax_debug:
        print(f"r{comm.Get_rank()}| Ar->encoded_gpu")

    return xla_client.ops.CustomCall(
        c,
        b"mpi_allreduce",
        operands=(
            x,
            token,
        ),
        shape=sh,
        opaque=descriptor,
        has_side_effect=True,
    )


# This function evaluates only the shapes during AST construction
def mpi_allreduce_abstract_eval(xs, token, op, comm, transpose):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        core.abstract_token,
    )


def mpi_allreduce_batch_eval(in_args, batch_axes, op, comm, transpose):
    x, token = in_args
    res = mpi_allreduce_p.bind(x, token, op=op, comm=comm, transpose=transpose)
    return res, batch_axes


def mpi_allreduce_value_and_jvp(in_args, tan_args, op, comm, transpose):
    x, token = in_args
    x_tan, token_tan = tan_args

    if mpi4jax_debug:
        print(f"r{comm.Get_rank()}| Ar->vjp scheduled for {x.shape}:{x.dtype} and {x_tan.shape}:{x_tan.dtype}")

    if unpack_hashable(op) != _MPI.SUM:
        raise NotImplementedError(
            "The adjoint of allreduce is only defined for op=MPI.SUM"
        )

    val, token = mpi_allreduce_p.bind(x, token, op=op, comm=comm, transpose=transpose)

    # throw away return token to work around jax#6285
    jvp, token_jvp = mpi_allreduce_p.bind(
        x_tan, token, op=op, comm=comm, transpose=transpose
    )
    return (val, token), (jvp, ad.Zero.from_value(token_jvp))


def mpi_allreduce_transpose_rule(tan_args, *x_args, op, comm, transpose):
    _, token = x_args
    x_tan, token_tan = tan_args

    if mpi4jax_debug:
        print(f"r{comm.Get_rank()}| Ar->transpose scheduled for {x_tan.shape}:{x_tan.dtype}")

    if unpack_hashable(op) != _MPI.SUM:
        raise NotImplementedError(
            "The linear transpose of allreduce is only defined for op=MPI.SUM"
        )

    res, token = mpi_allreduce_p.bind(
        x_tan, token, op=op, comm=comm, transpose=(not transpose)
    )
    return res, token_tan


mpi_allreduce_p.multiple_results = True
mpi_allreduce_p.def_impl(mpi_allreduce_impl)
register_abstract_eval(mpi_allreduce_p, mpi_allreduce_abstract_eval)

batching.primitive_batchers[mpi_allreduce_p] = mpi_allreduce_batch_eval

ad.primitive_jvps[mpi_allreduce_p] = mpi_allreduce_value_and_jvp
ad.primitive_transposes[mpi_allreduce_p] = mpi_allreduce_transpose_rule

# assign to the primitive the correct encoder
xla.backend_specific_translations["cpu"][mpi_allreduce_p] = mpi_allreduce_xla_encode_cpu
xla.backend_specific_translations["gpu"][mpi_allreduce_p] = mpi_allreduce_xla_encode_gpu
