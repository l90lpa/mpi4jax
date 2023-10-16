import numpy as _np
from mpi4py import MPI as _MPI

from jax import abstract_arrays, core
from jax.core import Primitive, Tracer, Token
from jax.interpreters import ad, batching
from jax.lax import create_token

from jax.interpreters import mlir
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
mpi_sendrecv_p = Primitive("sendrecv_mpi")  # Create the primitive
mpi_sendrecv_impl = default_primitive_impl(mpi_sendrecv_p)


# This function applies the primitive to an AST
@enforce_types(
    source=_np.integer,
    dest=_np.integer,
    sendtag=_np.integer,
    recvtag=_np.integer,
    comm=(type(None), _MPI.Intracomm, HashableMPIType),
    status=(type(None), _MPI.Status, HashableMPIType),
    # token=(type(None), Token, Tracer),
)
def sendrecv(
    sendbuf,
    recvbuf,
    token,
    source,
    dest,
    *,
    sendtag=0,
    recvtag=_MPI.ANY_TAG,
    comm=None,
    status=None,
):
    """Perform a sendrecv operation.

    .. warning::

        Unlike mpi4py's sendrecv, this returns a *new* array with the received data.

    Arguments:
        sendbuf: Array or scalar input to send.
        recvbuf: Array or scalar input with the correct shape and dtype. This can
           contain arbitrary data and will not be overwritten.
        token (Array): an array used in the same manor as an 'XLA token' to ensure correct execution order.
        source (int): Rank of the source MPI process.
        dest (int): Rank of the destination MPI process.
        sendtag (int): Tag of this message for sending.
        recvtag (int): Tag of this message for receiving.
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
        mpi_sendrecv_p.bind(
            sendbuf,
            recvbuf,
            token,
            source=source,
            dest=dest,
            sendtag=sendtag,
            recvtag=recvtag,
            comm=comm,
            status=status,
            _must_transpose=False,
        )
    )


# This function compiles the operation
@translation_rule_cpu
def mpi_sendrecv_xla_encode_cpu(
    ctx,
    sendbuf,
    recvbuf,
    token,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose=False,
):
    from ..xla_bridge.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR


    comm = unpack_hashable(comm)
    status = unpack_hashable(status)

    send_aval, recv_aval, *_ = ctx.avals_in
    send_nptype = send_aval.dtype
    recv_nptype = recv_aval.dtype

    send_type = ir.RankedTensorType(sendbuf.type)
    send_dtype = send_type.element_type
    send_dims = send_type.shape

    recv_type = ir.RankedTensorType(recvbuf.type)
    recv_dtype = recv_type.element_type
    recv_dims = recv_type.shape

    # compute total number of elements in arrays
    send_nitems = _np.prod(send_dims, dtype=int)
    send_dtype_handle = to_dtype_handle(send_nptype)

    recv_nitems = _np.prod(recv_dims, dtype=int)
    recv_dtype_handle = to_dtype_handle(recv_nptype)

    token_type = ir.RankedTensorType(token.type)
    token_dtype = token_type.element_type
    token_dims = token_type.shape

    if _must_transpose:
        out_types = [
            ir.RankedTensorType.get(send_dims, send_dtype),
            ir.RankedTensorType.get(token_dims, token_dtype),
        ]
    else: 
        out_types = [
            ir.RankedTensorType.get(recv_dims, recv_dtype),
            ir.RankedTensorType.get(token_dims, token_dtype),
        ]

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    if _must_transpose:
        operands = (
            as_mhlo_constant(recv_nitems, _np.intc),
            recvbuf,
            as_mhlo_constant(source, _np.intc),
            as_mhlo_constant(recvtag, _np.intc),
            as_mhlo_constant(recv_dtype_handle, _np.uintp),
            as_mhlo_constant(send_nitems, _np.intc),
            as_mhlo_constant(dest, _np.intc),
            as_mhlo_constant(sendtag, _np.intc),
            as_mhlo_constant(recv_dtype_handle, _np.uintp),
            as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
            as_mhlo_constant(status_ptr, _np.uintp),
            token,
        )
    else:
        operands = (
            as_mhlo_constant(send_nitems, _np.intc),
            sendbuf,
            as_mhlo_constant(dest, _np.intc),
            as_mhlo_constant(sendtag, _np.intc),
            as_mhlo_constant(send_dtype_handle, _np.uintp),
            as_mhlo_constant(recv_nitems, _np.intc),
            as_mhlo_constant(source, _np.intc),
            as_mhlo_constant(recvtag, _np.intc),
            as_mhlo_constant(recv_dtype_handle, _np.uintp),
            as_mhlo_constant(to_mpi_handle(comm), _np.uintp),
            as_mhlo_constant(status_ptr, _np.uintp),
            token,
        )

    return hlo_custom_call(
        b"mpi_sendrecv",
        out_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
    )


@translation_rule_gpu
def mpi_sendrecv_xla_encode_gpu(
    ctx,
    sendbuf,
    recvbuf,
    token,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose=False,
):


    from ..xla_bridge.mpi_xla_bridge import MPI_STATUS_IGNORE_ADDR
    from ..xla_bridge.mpi_xla_bridge_gpu import build_sendrecv_descriptor

    comm = unpack_hashable(comm)
    status = unpack_hashable(status)

    send_aval, recv_aval, *_ = ctx.avals_in
    send_nptype = send_aval.dtype
    recv_nptype = recv_aval.dtype

    send_type = ir.RankedTensorType(sendbuf.type)
    send_dtype = send_type.element_type
    send_dims = send_type.shape

    recv_type = ir.RankedTensorType(recvbuf.type)
    recv_dtype = recv_type.element_type
    recv_dims = recv_type.shape

    # compute total number of elements in arrays
    send_nitems = _np.prod(send_dims, dtype=int)
    send_dtype_handle = to_dtype_handle(send_nptype)

    recv_nitems = _np.prod(recv_dims, dtype=int)
    recv_dtype_handle = to_dtype_handle(recv_nptype)

    token_type = ir.RankedTensorType(token.type)
    token_dtype = token_type.element_type
    token_dims = token_type.shape

    if _must_transpose:
        out_types = [
            ir.RankedTensorType.get(send_dims, send_dtype),
            ir.RankedTensorType.get(token_dims, token_dtype),
        ]
    else: 
        out_types = [
            ir.RankedTensorType.get(recv_dims, recv_dtype),
            ir.RankedTensorType.get(token_dims, token_dtype),
        ]

    if status is None:
        status_ptr = _np.uintp(MPI_STATUS_IGNORE_ADDR)
    else:
        status_ptr = to_mpi_ptr(status)

    if _must_transpose:
        operands = (
            recvbuf,
            token,
        )
        descriptor = build_sendrecv_descriptor(
            recv_nitems,
            source,
            recvtag,
            recv_dtype_handle,
            send_nitems,
            dest,
            sendtag,
            send_dtype_handle,
            to_mpi_handle(comm),
            status_ptr,
        )
    else:
        operands = (
            sendbuf,
            token,
        )
        descriptor = build_sendrecv_descriptor(
            send_nitems,
            dest,
            sendtag,
            send_dtype_handle,
            recv_nitems,
            source,
            recvtag,
            recv_dtype_handle,
            to_mpi_handle(comm),
            status_ptr,
        )

    return hlo_custom_call(
        b"mpi_sendrecv",
        out_types=out_types,
        operands=operands,
        operand_layouts=get_default_layouts(operands),
        result_layouts=get_default_layouts(out_types),
        has_side_effect=True,
        backend_config=descriptor,
    )


# This function evaluates only the shapes during AST construction
def mpi_sendrecv_abstract_eval(
    sendbuf,
    recvbuf,
    token,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose=False,
):
    return (
        abstract_arrays.ShapedArray(recvbuf.shape, recvbuf.dtype),
        abstract_arrays.ShapedArray(token.shape, token.dtype),
    ), {effect}


def mpi_sendrecv_batch_eval(
    in_args,
    batch_axes,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose=False,
):

    sendbuf, recvbuf, token = in_args

    assert batch_axes[0] == batch_axes[1]

    res = mpi_sendrecv_p.bind(
        sendbuf,
        recvbuf,
        token,
        source=source,
        dest=dest,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
        _must_transpose=_must_transpose,
    )
    return res, (batch_axes[0], batch_axes[2])


def mpi_sendrecv_value_and_jvp(
    in_args,
    tan_args,
    source,
    dest,
    sendtag,
    recvtag,
    comm,
    status,
    _must_transpose=False,
):
    sendbuf, recvbuf, token = in_args
    send_tan, recv_tan, token_tan = tan_args

    val, token = mpi_sendrecv_p.bind(
        sendbuf,
        recvbuf,
        token,
        source=source,
        dest=dest,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
        _must_transpose=_must_transpose,
    )

    if isinstance(token_tan, ad.Zero):
        token_tan = token
    else:
        token_tan = token_tan + token
    jvp, token_tan = mpi_sendrecv_p.bind(
        send_tan,
        recv_tan,
        token_tan,
        source=source,
        dest=dest,
        sendtag=sendtag,
        recvtag=recvtag,
        comm=comm,
        status=status,
        _must_transpose=_must_transpose,
    )

    return (val, token), (jvp, token_tan)


def mpi_sendrecv_transpose_rule(
    tan_args, *x_args, source, dest, sendtag, recvtag, comm, status, _must_transpose
):
    _, _, token = x_args
    out_tan, token_tan = tan_args

    if isinstance(token_tan, ad.Zero):
        if isinstance(token, ad.UndefinedPrimal):
            token_tan = jnp.zeros(token.aval.shape, token.aval.dtype)
        else:
            token_tan = jnp.zeros(token.shape, token.dtype)
    # swap the sender and receiver
    res, token_tan = mpi_sendrecv_p.bind(
        out_tan,
        out_tan,
        token_tan,
        source=dest,
        dest=source,
        sendtag=common_mpi_send_recv_vjp_tag,
        recvtag=common_mpi_send_recv_vjp_tag,
        comm=comm,
        status=status,
        _must_transpose=_must_transpose,
    )
    return res, ad.Zero.from_value(res), token_tan


mpi_sendrecv_p.multiple_results = True
mpi_sendrecv_p.def_impl(mpi_sendrecv_impl)
mpi_sendrecv_p.def_effectful_abstract_eval(mpi_sendrecv_abstract_eval)

batching.primitive_batchers[mpi_sendrecv_p] = mpi_sendrecv_batch_eval

ad.primitive_jvps[mpi_sendrecv_p] = mpi_sendrecv_value_and_jvp
ad.primitive_transposes[mpi_sendrecv_p] = mpi_sendrecv_transpose_rule

# assign to the primitive the correct encoder
mlir.register_lowering(mpi_sendrecv_p, mpi_sendrecv_xla_encode_cpu, platform="cpu")
mlir.register_lowering(mpi_sendrecv_p, mpi_sendrecv_xla_encode_gpu, platform="cuda")
