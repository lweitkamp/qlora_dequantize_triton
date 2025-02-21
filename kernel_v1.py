import torch
import triton
import triton.language as tl

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
def dequantize_kernel_v1(
    weights_ptr, weights_quant_ptr, weights_code_ptr,
    absmax_quant_ptr, absmax_code_ptr, absmax_scale_ptr, absmax_mean,
    blocksize_weights, blocksize_absmax, numel_weights,
    BLOCK_SIZE: tl.constexpr,
):
    numel_absmax = numel_weights // blocksize_weights

    input_offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offset < numel_weights

    # Retrieve the quantized absmax values and cast to int (lookup indices).
    absmax_offset = input_offset // blocksize_weights
    absmax_mask = absmax_offset < numel_absmax
    absmax_quant = tl.load(absmax_quant_ptr + absmax_offset, mask=absmax_mask).to(tl.int16)

    # Every blocksize_absmax part of absmax is scaled with the same value.
    absmax_scale_offset = absmax_offset // blocksize_absmax
    absmax_scale_mask = absmax_offset < numel_absmax
    absmax_scale = tl.load(
        absmax_scale_ptr + absmax_scale_offset, mask=absmax_scale_mask
    )

    # Lookup the codepoint for each absmax index and scale appropriately.
    absmax_code = tl.load(absmax_code_ptr + absmax_quant, mask=absmax_mask)
    absmax = absmax_code * absmax_scale + absmax_mean

    # Load the quantized uint8 weights.
    # Weights are contiguous, we could opt to compile hint as such
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


def dequantize_triton_v1(quant_weights, quant_state):
    dtype_in = dtype_out = quant_state.dtype
    if dtype_in == torch.bfloat16:
        device = torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(device)
        dtype_in = torch.float32 if major < 8 else torch.bfloat16

    weights = torch.empty(
        quant_state.shape, dtype=dtype_in, device=quant_weights.device
    )

    weights_code = NF4_DEQUANT_TABLE_TRITON.to(device=quant_weights.device)

    launch_grid = lambda meta: (triton.cdiv(quant_weights.numel(), meta["BLOCK_SIZE"]),)
    dequantize_kernel_v1[launch_grid](
        weights_ptr=weights,
        weights_quant_ptr=quant_weights,
        weights_code_ptr=weights_code,
        absmax_quant_ptr=quant_state.absmax,
        absmax_code_ptr=quant_state.state2.code,
        absmax_scale_ptr=quant_state.state2.absmax,
        absmax_mean=quant_state.offset.item(),
        numel_weights=quant_weights.numel(),
        blocksize_weights=quant_state.blocksize // 2,
        blocksize_absmax=quant_state.state2.blocksize,
        BLOCK_SIZE=max(512, min(32, quant_weights.numel() // 256)),
    )

    return weights.to(dtype_out)
