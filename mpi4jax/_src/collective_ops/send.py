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
mpi_send_p = Primitive("send_mpi")  # Create the primitive
mpi_send_impl = default_primitive_impl(mpi_send_p)


# This function applies the primitive to an AST
@enforce_types(
    dest=_np.integer,
    tag=_np.integer,
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    # token=(type(None), Token, Tracer),
)
def send(x, token, dest, *, tag=0, comm=None):
    """Perform a send operation.

    Arguments:
        x: Array or scalar input to send.
        token (Array): an array used in the same manor as an 'XLA token' to ensure correct execution order.
        dest (int): Rank of the destination MPI process.
        tag (int): Tag of this message.
        comm (mpi4py.MPI.Comm): The MPI communicator to use (defaults to
            a clone of :obj:`COMM_WORLD`).

    Returns:
        Tuple[Token]:
            - A new, modified token, that depends on this operation.

    """

    if comm is None:
        comm = get_default_comm()

    comm = wrap_as_hashable(comm)
    return mpi_send_p.bind(x, token, dest=dest, tag=tag, comm=comm)


# This function compiles the operation
@translation_rule_cpu
def mpi_send_xla_encode_cpu(ctx, x, token, dest, tag, comm):
    comm = unpack_hashable(comm)

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
        ir.RankedTensorType.get(token_dims, token_dtype),
    ]

    operands = (
        as_mhlo_constant(nitems, _np.intc),
        x,
        as_mhlo_constant(dest, _np.intc),
        as_mhlo_constant(tag, _np.intc),
        as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
        as_mhlo_constant(dtype_handle, _np.uintp),
        token,
    )

    # JAX insists on outputs being iterable
    return hlo_custom_call(
            b"mpi_send",
            out_types=out_types,
            operands=operands,
            operand_layouts=get_default_layouts(operands),
            result_layouts=get_default_layouts(out_types),
            has_side_effect=True,
        )


@translation_rule_gpu
def mpi_send_xla_encode_gpu(ctx, x, token, dest, tag, comm):
    from ..xla_bridge.mpi_xla_bridge_gpu import build_send_descriptor

    comm = unpack_hashable(comm)

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
        ir.RankedTensorType.get(token_dims, token_dtype),
    ]

    operands = (
        x,
        token,
    )

    descriptor = build_send_descriptor(
        nitems,
        dest,
        tag,
        to_mpi_handle(comm),
        dtype_handle,
    )

    # JAX insists on outputs being iterable
    return hlo_custom_call(
            b"mpi_send",
            out_types=out_types,
            operands=operands,
            operand_layouts=get_default_layouts(operands),
            result_layouts=get_default_layouts(out_types),
            has_side_effect=True,
            backend_config=descriptor,
        )


# This function evaluates only the shapes during AST construction
def mpi_send_abstract_eval(xs, token, dest, tag, comm):
    return abstract_arrays.ShapedArray(token.shape, token.dtype), {effect}

def mpi_send_value_and_jvp(primal_args, tangent_args, dest, tag, comm):
    
    sendbuf, token = primal_args
    sendbuf_tan, token_tan = tangent_args

    token = mpi_send_p.bind(sendbuf, token, dest=dest, tag=tag, comm=comm)

    token_tan = token_tan + token
    token_tan = mpi_send_p.bind(sendbuf_tan, token_tan, dest=dest, tag=tag, comm=comm)

    return token, token_tan


def mpi_send_transpose_rule(cotan_args, *primal_args, dest, tag, comm):
    from .recv import mpi_recv_p

    sendbuf, token = primal_args
    token_cot = cotan_args

    if isinstance(sendbuf, ad.UndefinedPrimal):
        zero_cot = jnp.zeros(sendbuf.aval.shape, sendbuf.aval.dtype)
    else:
        zero_cot = jnp.zeros(sendbuf.shape, sendbuf.dtype)
    sendbuf_cot, token_cot = mpi_recv_p.bind(zero_cot, token_cot, source=dest, tag=common_mpi_send_recv_vjp_tag, comm=comm, status=None)

    return sendbuf_cot, token_cot


mpi_send_p.def_impl(mpi_send_impl)
mpi_send_p.def_effectful_abstract_eval(mpi_send_abstract_eval)

ad.primitive_jvps[mpi_send_p] = mpi_send_value_and_jvp
ad.primitive_transposes[mpi_send_p] = mpi_send_transpose_rule

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_send_p, mpi_send_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_send_p, mpi_send_xla_encode_gpu, platform="cuda")
