import numpy as _np
from mpi4py import MPI as _MPI

from jax import abstract_arrays, core
from jax.core import Primitive, Tracer, Token
from jax.lax import create_token

from jax.interpreters import mlir, ad
import jaxlib.mlir.ir as ir
import jax.numpy as jnp

from ..utils import (
    HashableMPIType,
    default_primitive_impl,
    to_dtype_handle,
    to_mpi_handle,
    to_mpi_ptr,
    unpack_hashable,
    wrap_as_hashable,
    as_mhlo_constant,
    get_default_layouts,
    effect,
    common_mpi_send_recv_vjp_tag,
)
from ..jax_compat import hlo_custom_call, token_type
from ..decorators import translation_rule_cpu, translation_rule_gpu
from ..validation import enforce_types
from ..comm import get_default_comm


# The Jax primitive
mpi_recv_p = Primitive("recv_mpi")  # Create the primitive
mpi_recv_impl = default_primitive_impl(mpi_recv_p)


# This function applies the primitive to an AST
@enforce_types(
    source=_np.integer,
    tag=_np.integer,
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    status=(type(None), _MPI.Status, HashableMPIType),
    # token=(type(None), Token, Tracer),
)
def recv(
    x,
    token,
    source=_MPI.ANY_SOURCE,
    *,
    tag=_MPI.ANY_TAG,
    comm=None,
    status=None,
):
    """Perform a recv (receive) operation.

    .. warning::

        Unlike mpi4py's recv, this returns a *new* array with the received data.

    Arguments:
        x: Array or scalar input with the correct shape and dtype. This can contain
           arbitrary data and will not be overwritten.
        token (Array): an array used in the same manor as an 'XLA token' to ensure correct execution order.
        source (int): Rank of the source MPI process.
        tag (int): Tag of this message.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).
        status (mpi4py.MPI.Status): Status object, can be used for introspection.

    Returns:
        Tuple[DeviceArray, Token]:
            - Received data.
            - A new, modified token, that depends on this operation.

    """

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)

    if status is not None:
        status = wrap_as_hashable(status)

    return tuple(
        mpi_recv_p.bind(x, token, source=source, tag=tag, comm=comm, status=status)
    )


# This function compiles the operation
@translation_rule_cpu
def mpi_recv_xla_encode_cpu(ctx, x, token, source, tag, comm, status):
    from ..xla_bridge.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR

    comm = unpack_hashable(comm)
    status = unpack_hashable(status)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    token_type = ir.RankedTensorType(token.type)
    token_dtype = token_type.element_type
    token_dims = token_type.shape

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        ir.RankedTensorType.get(token_dims, token_dtype),
    ]

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    operands = (
        as_mhlo_constant(nitems, _np.intc),
        as_mhlo_constant(source, _np.intc),
        as_mhlo_constant(tag, _np.intc),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        as_mhlo_constant(dtype_handle, _np.uintp),
        as_mhlo_constant(status_ptr, _np.uintp),
        token,
    )

    return hlo_custom_call(
        b"mpi_recv",
        out_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
    )


@translation_rule_gpu
def mpi_recv_xla_encode_gpu(ctx, x, token, source, tag, comm, status):
    from ..xla_bridge.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR
    from ..xla_bridge.mpi_xla_bridge_gpu import build_recv_descriptor

    comm = unpack_hashable(comm)
    status = unpack_hashable(status)

    x_aval, *_ = ctx.avals_in
    x_nptype = x_aval.dtype

    x_type = ir.RankedTensorType(x.type)
    dtype = x_type.element_type
    dims = x_type.shape

    # compute total number of elements in array
    nitems = _np.prod(dims, dtype=int)
    dtype_handle = to_dtype_handle(x_nptype)

    token_type = ir.RankedTensorType(token.type)
    token_dtype = token_type.element_type
    token_dims = token_type.shape

    out_types = [
        ir.RankedTensorType.get(dims, dtype),
        ir.RankedTensorType.get(token_dims, token_dtype),
    ]

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    operands = (token,)

    descriptor = build_recv_descriptor(
        nitems,
        source,
        tag,
        to_mpi_handle(comm),
        dtype_handle,
        status_ptr,
    )

    return hlo_custom_call(
        b"mpi_recv",
        out_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
        backend_config=descriptor,
    )


# This function evaluates only the shapes during AST construction
def mpi_recv_abstract_eval(xs, token, source, tag, comm, status):
    return (
        abstract_arrays.ShapedArray(xs.shape, xs.dtype),
        abstract_arrays.ShapedArray(token.shape, token.dtype),
    ), {effect}


def mpi_recv_value_and_jvp(primal_args, tangent_args, source, tag, comm, status):

    recvbuf, token = primal_args
    recvbuf_tan, token_tan = tangent_args

    recvbuf, token = mpi_recv_p.bind(recvbuf, token, source=source, tag=tag, comm=comm, status=status)

    token_tan = token_tan + token
    recvbuf_tan, token_tan = mpi_recv_p.bind(recvbuf_tan, token_tan, source=source, tag=tag, comm=comm, status=status)

    return (recvbuf, token), (recvbuf_tan, token_tan)


def mpi_recv_transpose_rule(cotan_args, *primal_args, source, tag, comm, status):
    from .send import mpi_send_p

    recvbuf, token = primal_args
    recvbuf_cot, token_cot = cotan_args

    token_cot = mpi_send_p.bind(recvbuf_cot, token_cot, dest=source, tag=common_mpi_send_recv_vjp_tag, comm=comm)

    return jnp.zeros_like(recvbuf_cot), token_cot


mpi_recv_p.multiple_results = True
mpi_recv_p.def_impl(mpi_recv_impl)
mpi_recv_p.def_effectful_abstract_eval(mpi_recv_abstract_eval)

ad.primitive_jvps[mpi_recv_p] = mpi_recv_value_and_jvp
ad.primitive_transposes[mpi_recv_p] = mpi_recv_transpose_rule

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_recv_p, mpi_recv_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_recv_p, mpi_recv_xla_encode_gpu, platform="cuda")
