# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import torch
'''
def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=torch.bool)
    return torch.tril(ret)
'''


def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask


def make_length_attn_mask(xs_lens):
    """
    xs_lens = torch.tensor([14,15,16,10,8])
    生成 【5，16，16】的矩阵，按照lens 进行msk
    """
    # 获取最大长度
    max_len = xs_lens.max()
    
    # 生成行索引和列索引，形状为 (1, max_len, 1) 和 (1, 1, max_len)
    row_indices = torch.arange(max_len, device=xs_lens.device).view(1, -1, 1)
    col_indices = torch.arange(max_len, device=xs_lens.device).view(1, 1, -1)
    
    # 将 xs_lens 扩展为 (bs, 1, 1)
    xs_lens = xs_lens.view(-1, 1, 1)
    
    # 利用广播机制生成掩码
    mask = (row_indices < xs_lens) & (col_indices < xs_lens)
    
    return mask.bool()

def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        start = int(start)
        ending = int(ending)

        ret[i, start:ending] = True
    return ret



def get_masks_len(mask):
    # 确保 mask 是布尔类型（True/False）
    if not mask.is_floating_point():
        mask = mask.bool()
    
    # 展平每个样本为一维张量（形状变为 [5, 16*16]）
    flat_mask = mask.view(mask.size(0), -1)
    
    # 计算每个样本中 True 的个数（形状为 [5]）
    true_counts = flat_mask.sum(dim=1)
    
    # 调整为 [5, 1] 的形状
    return true_counts.unsqueeze(1)


def add_optional_chunk_mask(xs: torch.Tensor,
                            masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            enable_full_context: bool = True):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context (bool):
            True: chunk size is either [1, 25] or full context(max_len)
            False: chunk size ~ U[1, 25]

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """

              
    mask_lens = get_masks_len(masks)
    #print('mask_lens', mask_lens)
    mask_expand = make_length_attn_mask(mask_lens)

    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            #print('static_chunk_size', static_chunk_size)
            chunk_size = static_chunk_size
            if use_dynamic_left_chunk:
                max_left_chunks = (max_len - 1) // chunk_size
                #max_left_chunks = min(max_left_chunks, 10)
                num_left_chunks = torch.randint(1, max_left_chunks, (1, )).item()
                #print('=====num_left_chunks', num_left_chunks, 'chunk_size', chunk_size)
            
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = mask_expand & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks_tmp = chunk_masks
        #print('static_chunk_size chunk_masks', chunk_masks)
        
        chunk_masks = mask_expand & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    assert chunk_masks.dtype == torch.bool
    

    #torch.set_printoptions(profile="full") 
   
    if (chunk_masks.sum(dim=-1) == 0).sum().item() != 0:
        # streaming模式下，chunk_masks 全部是false，
        # chunk_mask 是尾部，length_mask 是头部，交叉为0
        # 这种样本不要了，全部mask
        '''
        print('chunk_masks_tmp', chunk_masks_tmp)
        print('masks', masks)
        
        
        masks_len=torch.sum(masks, dim=2)
        print('mask', masks_len)
        chunk_masks_tmp_len=torch.sum(chunk_masks_tmp, dim=-1)
        print('chunk_masks_tmp', chunk_masks_tmp_len)
        print('chunk_masks_tmp', chunk_masks_tmp)

        print('xs', xs.size())
        print('use_dynamic_chunk', use_dynamic_chunk)
        print('use_dynamic_left_chunk', use_dynamic_left_chunk)
        print('decoding_chunk_size', decoding_chunk_size)
        print('static_chunk_size', static_chunk_size)
        print('num_decoding_left_chunks', num_decoding_left_chunks)
        print('enable_full_context', enable_full_context)
        '''
        #print('get chunk_masks all false at some timestep, force set to true, make sure they are masked in futuer computation!')
        chunk_masks[chunk_masks.sum(dim=-1) == 0] = False
        #sys.exit()
        
    
    return chunk_masks

def add_optional_chunk_mask_old(xs: torch.Tensor,
                            masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            enable_full_context: bool = True):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context (bool):
            True: chunk size is either [1, 25] or full context(max_len)
            False: chunk size ~ U[1, 25]

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """

              
    

    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = torch.randint(1, max_len, (1, )).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                #chunk_size = chunk_size % 25 + 1
                chunk_size = chunk_size % static_chunk_size + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks,
                                                    (1, )).item()
                    print('=====num_left_chunks', num_left_chunks, 'chunk_size', chunk_size)
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks_tmp = chunk_masks
        #print('static_chunk_size chunk_masks', chunk_masks)
        
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    assert chunk_masks.dtype == torch.bool
    

    #torch.set_printoptions(profile="full") 
   
    if (chunk_masks.sum(dim=-1) == 0).sum().item() != 0:
        # streaming模式下，chunk_masks 全部是false，
        # chunk_mask 是尾部，length_mask 是头部，交叉为0
        # 这种样本不要了，全部mask
        '''
        print('chunk_masks_tmp', chunk_masks_tmp)
        print('masks', masks)
        
        
        masks_len=torch.sum(masks, dim=2)
        print('mask', masks_len)
        chunk_masks_tmp_len=torch.sum(chunk_masks_tmp, dim=-1)
        print('chunk_masks_tmp', chunk_masks_tmp_len)
        print('chunk_masks_tmp', chunk_masks_tmp)

        print('xs', xs.size())
        print('use_dynamic_chunk', use_dynamic_chunk)
        print('use_dynamic_left_chunk', use_dynamic_left_chunk)
        print('decoding_chunk_size', decoding_chunk_size)
        print('static_chunk_size', static_chunk_size)
        print('num_decoding_left_chunks', num_decoding_left_chunks)
        print('enable_full_context', enable_full_context)
        '''
        #print('get chunk_masks all false at some timestep, force set to true, make sure they are masked in futuer computation!')
        
        chunk_masks[chunk_masks.sum(dim=-1) == 0] = True
        #sys.exit()
    
    return chunk_masks


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask
