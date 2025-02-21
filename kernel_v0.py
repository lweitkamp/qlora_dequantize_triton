import torch
import triton
import triton.language as tl
from torch import Tensor

# Lookup table for 4-bit dequantization
NF4_DEQUANT_TABLE_TRITON = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)


@triton.jit
def dequantize_absmax_kernel_v0(
    absmax_ptr, absmax_quant_ptr, absmax_code_ptr, absmax_scale_ptr,
    numel, blocksize,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Retrieve the quantized absmax values and cast to int.
    absmax_quant = tl.load(absmax_quant_ptr + offsets, mask=mask).to(tl.int32)
    absmax_code = tl.load(absmax_code_ptr + absmax_quant, mask=mask)

    # Every blocksize_absmax part of absmax is scaled with the same value.
    scale_offsets = offsets // blocksize
    scale_mask = scale_offsets < (numel // blocksize)
    absmax_scale = tl.load(absmax_scale_ptr + scale_offsets, mask=scale_mask)

    # Dequantize absmax.
    absmax = absmax_code * absmax_scale

    # Store the dequantized values into the output buffer.
    tl.store(absmax_ptr + offsets, absmax, mask=offsets < numel)


@triton.jit
def dequantize_weights_kernel_v0(
    weights_ptr, weights_quant_ptr, weights_code_ptr,
    absmax_ptr,
    numel_weights: int, blocksize: int,
    BLOCK_SIZE: tl.constexpr,
):
    input_offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offset < numel_weights

    # Load the absmax value for this block.
    absmax_offset = input_offset // blocksize
    absmax_mask = absmax_offset < numel_weights
    absmax = tl.load(absmax_ptr + absmax_offset, mask=absmax_mask)

    # Load the quantized uint8 weights.
    weights_quant_offset = tl.max_contiguous(
        tl.multiple_of(input_offset, BLOCK_SIZE), BLOCK_SIZE
    )
    weights_quant = tl.load(weights_quant_ptr + weights_quant_offset, mask=input_mask)

    # Unpack first and last four bits, look up corresponding value in code table.
    first_four = weights_quant >> 4
    weights_first = (
        tl.load(weights_code_ptr + first_four, mask=first_four < 16) * absmax
    )
    last_four = weights_quant & 0x0F
    weights_second = tl.load(weights_code_ptr + last_four, mask=last_four < 16) * absmax

    # Compute output positions - the values will be stored interleaved (not great).
    out_offsets_first = input_offset * 2
    out_offsets_last = input_offset * 2 + 1
    tl.store(
        weights_ptr + out_offsets_first,
        weights_first,
        mask=out_offsets_first < numel_weights * 2,
    )
    tl.store(
        weights_ptr + out_offsets_last,
        weights_second,
        mask=out_offsets_last < numel_weights * 2,
    )


def dequantize_triton_v0(quant_weights: Tensor, quant_state: Tensor) -> Tensor:
    dtype_in = dtype_out = quant_state.dtype
    if dtype_in == torch.bfloat16:
        device = torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(device)
        dtype_in = torch.float32 if major < 8 else torch.bfloat16

    numel_absmax = quant_state.absmax.numel()
    numel_weights = quant_weights.numel()

    # Absmax is always fp32.
    absmax = torch.empty(numel_absmax, dtype=torch.float32, device=quant_weights.device)

    launch_grid = lambda meta: (triton.cdiv(numel_absmax, meta["BLOCK_SIZE"]),)
    dequantize_absmax_kernel_v0[launch_grid](
        absmax_ptr=absmax,
        absmax_quant_ptr=quant_state.absmax,
        absmax_code_ptr=quant_state.state2.code,
        absmax_scale_ptr=quant_state.state2.absmax,
        numel=numel_absmax,
        blocksize=quant_state.state2.blocksize,
        BLOCK_SIZE=max(512, min(32, numel_absmax // 256)),
    )

    absmax += quant_state.offset

    weights = torch.empty(
        quant_state.shape, dtype=dtype_in, device=quant_weights.device
    )

    weights_code = NF4_DEQUANT_TABLE_TRITON.to(device=quant_weights.device)

    launch_grid = lambda meta: (triton.cdiv(numel_weights, meta["BLOCK_SIZE"]),)
    dequantize_weights_kernel_v0[launch_grid](
        weights_ptr=weights,
        weights_quant_ptr=quant_weights,
        weights_code_ptr=weights_code,
        absmax_ptr=absmax,
        numel_weights=numel_weights,
        blocksize=quant_state.blocksize // 2,
        BLOCK_SIZE=max(512, min(32, numel_weights // 256)),
    )

    return weights.to(dtype_out)
