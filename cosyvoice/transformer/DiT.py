import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
from matcha.models.components.decoder import SinusoidalPosEmb, Block1D, ResnetBlock1D, Downsample1D, Upsample1D
from einops import pack, rearrange, repeat
from cosyvoice.utils.mask import add_optional_chunk_mask
from cosyvoice.utils.common import mask_to_bias
from cosyvoice.transformer.dit_modules import DiTBlock, TimestepEmbedding, AdaLayerNormZero_Final, RotaryEmbedding
import os
import pickle as pkl
def save_pikles(data, name):
    if False:
        return 
    save_base = os.path.join(os.path.dirname(__file__), "../../", "pickles_save")
    os.makedirs(save_base, exist_ok=True)
    save_name = os.path.join(save_base, name)
    with open(save_name, 'wb') as f:
        pkl.dump(data.detach().cpu(), f)




class DiT(nn.Module):
    """Single DiT Block (Figure 1-left)"""
    def __init__(
        self,
        in_channels,
        out_channels,
        att_dim=256,
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snake",
        static_chunk_size=50,
        num_decoding_left_chunks=2,
    ):
        """
        This decoder requires an input with the same shape of the target. So, if your text content
        is shorter or longer than the outputs, please re-sampling it before feeding to the decoder.
        """
        torch.nn.Module.__init__(self)
        self.in_channels = in_channels  # 240
        self.out_channels = out_channels # 80
        self.time_embd = TimestepEmbedding(att_dim)
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks
        output_channel = in_channels

        self.rotary_embed = RotaryEmbedding(att_dim // num_heads)
        # print('self.rotary : {}'.format(self.rotary))

        self.input_proj = nn.Linear(in_channels, att_dim)
        self.attn_processor = "stream_block_sr_L"
        self.dit_blocks = nn.ModuleList()
        if self.attn_processor == "stream_block_sr_L":
            attn_processor_0 = 'stream_block_sr_00'
            attn_processor_1 = 'stream_block_sr_10'
            attn_processor_2 = 'stream_block_sr_01'
            for i in range(n_blocks):
                if i == 0 or i == n_blocks // 2 or i == n_blocks-1:
                    attn_processor_in = attn_processor_1
                elif i == n_blocks // 3: 
                    attn_processor_in = attn_processor_2
                else:
                    attn_processor_in = attn_processor_0
                self.dit_blocks.append(
                    DiTBlock(
                        dim=att_dim, heads=num_heads, dim_head=att_dim//num_heads, 
                        ff_mult=4, dropout=0.0, attn_processor=attn_processor_in
                    )
                )
        self.norm_out = AdaLayerNormZero_Final(att_dim)  # final modulation
        self.final_proj = nn.Linear(att_dim, self.out_channels)

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        t = self.time_embd(t)    # [bsz, seq_len]

        conds = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            conds = pack([conds, spks], "b * t")[0]
        if cond is not None:
            conds = pack([conds, cond], "b * t")[0]  # [bsz, 320, seq_len * 2]
        # print('conds ori = {}'.format(conds.shape))
        x = conds 
        t_cond = t 

        # conds.shape = [bsz, 240, len * 2]
        seq_len = x.shape[-1]
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        # print('rotary_pos : {}'.format(rotary_pos.shape))

        x = rearrange(x, "b c t -> b t c").contiguous()           # x.shape = [bsz, len * 2, 80]

        if streaming is True:
            attn_mask = add_optional_chunk_mask(x, mask.bool(), False, False, 0, self.static_chunk_size, self.num_decoding_left_chunks)
        else:
            attn_mask = add_optional_chunk_mask(x, mask.bool(), False, False, 0, 0, -1).repeat(1, x.size(1), 1)
        attn_mask = mask_to_bias(attn_mask, x.dtype).bool().logical_not()

        x = self.input_proj(x)    # [bsz, len * 2, 256]
        for dit_block in self.dit_blocks:
            # save_pikles(mask.squeeze(1), "input_mask")
            # print("input mask : {}".format(mask.squeeze(1).bool()))
            # mask : 
            # [t, t, t, f, f, ]
            # [t, t, t, t, t, ]
            x = dit_block(x, t_cond, mask=mask.squeeze(1).bool(), attn_mask=attn_mask, rope=rope)
        x = self.norm_out(x, t_cond)
        x = self.final_proj(x)    # [bsz, len * 2, 80]
        x = x.transpose(-2, -1) 
        return x * mask
    
    def forward_chunk(self, x, mask, mu, t, spks=None, cond=None, streaming=False, cache=None):
        t = self.time_embd(t)    # [bsz, seq_len]
        conds = pack([x, mu], "b * t")[0]
        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            conds = pack([conds, spks], "b * t")[0]
        if cond is not None:
            conds = pack([conds, cond], "b * t")[0]  # [bsz, 320, seq_len * 2]
        # print('conds ori = {}'.format(conds.shape))
        x = conds 
        t_cond = t

        # conds.shape = [bsz, 240, len * 2]
        seq_len = x.shape[-1]
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        # print('rotary_pos : {}'.format(rotary_pos.shape))

        x = rearrange(x, "b c t -> b t c").contiguous()           # x.shape = [bsz, len * 2, 80]

        attn_mask = torch.ones(x.size(0), x.size(1), x.size(1), device=x.device).bool()
        attn_mask = mask_to_bias(attn_mask, x.dtype).bool().logical_not()

        x = self.input_proj(x)    # [bsz, len * 2, 256]
        for dit_block in self.dit_blocks:
            x = dit_block(x, t_cond, mask=mask.squeeze(1).bool(), attn_mask=attn_mask, rope=rope)
        x = self.norm_out(x, t_cond)
        x = self.final_proj(x)    # [bsz, len * 2, 80]
        x = x.transpose(-2, -1) 
        return x * mask, None, None, None, None, None, None, None




class FFN(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # 两个输入线性层
        intermediate_dim = int(2 / 3 * 4 * hidden_dim)
        self.w1 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        # 输出线性层
        self.w3 = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


def test_dit():
    # 设置随机种子以便复现结果
    torch.manual_seed(42)
    
    # 定义模型参数
    batch_size = 4
    seq_len = 32
    hidden_dim = 80
    num_heads = 8


    # 初始化DiT模块
    dit = DiT(in_channels=hidden_dim,
                out_channels=hidden_dim,
                channels=[256],
                dropout=0.0,
                attention_head_dim=10,
                n_blocks=4,
                num_mid_blocks=12,
                num_heads=num_heads,
                act_fn="gelu",
                static_chunk_size=0.4 * 25 * 2,
                num_decoding_left_chunks=1)

    # 生成随机输入数据
    x = torch.randn(batch_size, seq_len, hidden_dim)  # 主输入
    text_emb = torch.randn(batch_size, seq_len, hidden_dim)  # 文本嵌入
    time_emb = torch.randn(batch_size, hidden_dim)  # 时间嵌入
    
    # 生成旋转位置编码
    rotary = RotaryEmbedding(dim=hidden_dim // num_heads)
    rotary_pos = rotary(seq_len, x.device)
    
    # 打印输入形状
    # print(f"输入形状:")
    # print(f"x: {x.shape}")
    # print(f"text_emb: {text_emb.shape}")
    # print(f"time_emb: {time_emb.shape}")
    # print(f"rotary_pos: {rotary_pos.shape}")
    
    # 前向传播
    output = dit(x, text_emb, time_emb, rotary_pos)
    
    # 打印输出形状
    # print(f"\n输出形状: {output.shape}")
    
    # 检查输出是否合理
    assert output.shape == x.shape, "输出形状应与输入形状一致"
    # print("\n测试通过: 输出形状正确")

if __name__ == "__main__":
    test_dit()
