import math

import inspect
from collections import defaultdict
from typing import Any, Callable, Optional

from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import bitsandbytes as bnb

from lib.tokenizer.base import BaseTokenizer


"""
References:
1) The official GPT-2 TensorFlow implementation released by OpenAI: `https://github.com/openai/gpt-2/blob/master/src/model.py`
2) `huggingface/transformers` PyTorch implementation: `https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py`
3) Nano GPT: `https://github.com/karpathy/nanoGPT/blob/master/model.py`
4) ALiBi
6) Fast Transformer Decoding / Multi-Query Attention: `https://arxiv.org/pdf/1911.02150v1.pdf`
7) Compute-Optimal Training: `https://arxiv.org/pdf/2203.15556.pdf`, `https://tomekkorbak.com/2022/10/10/compute-optimal-gpt2/`
"""

# https://github.com/kyegomez/AttentionIsOFFByOne
# Define the softmax_one function with added one in the denominator, which helps to reduce
#   the negative impact of tiny values in the softmax function and improves numerical stability
def softmax_one(x, dim=None, _stacklevel=3, dtype=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))


class ALiBi:
    @staticmethod
    def bias(n, device=None):
        bias = torch.zeros(n, n)
        for i in range(n):
            bias[i, :i] = -torch.arange(i, 0, -1)
        return bias

    @staticmethod
    def get_slopes(n, ignore_workaround=True):
        """
        `ignore_workaround=False` means alternate slopes are calculated as per. Facebook/Meta AI.
        - When workaround is in effect (not ignored), numbers come out 'rounder', e.g. .5, .25 etc.
        """
        # ALiBi: We do not add position embeddings at any point in the network. The only
        # modification we apply is after the query-key dot product, where we add a static, non-learned bias:
        # softmax(q(i)K.T + m @ [-(i - 1), ..., -2, -1, 0]),
        # where scalar `m` is a head-specific slope fixed before training.
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(n).is_integer() or ignore_workaround:
            return get_slopes_power_of_2(n)                   # In the paper, we only train models that have 2^a heads for some a. This function has
        else:                                                 # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2**math.floor(math.log2(n))  # when the number of heads is not a power of 2, we use this workaround.

            # 2^(floor(log2(12))) = 8

            return get_slopes_power_of_2(closest_power_of_2) + ALiBi.get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalAttention(nn.Module):
    def __init__(self, config, c_kv=None, alibi_params=None, use_cache=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0  # Because we unpack N heads across the embedding.

        # Shared KV for Multi-query Attention, or per-block if not supplied.
        self.c_kv = c_kv if c_kv else nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)

        self.cache_k = None
        self.cache_v = None

        self.use_cache = use_cache

        # Causal `query` value per-head.
        self.c_query = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # self.alibi_bias = None
        # self.alibi_bias_T = None

        self.alibi_offset = alibi_params['alibi_offset']
        self.alibi_m = alibi_params['alibi_m']
        self.c_mask = alibi_params['c_mask']

        # Output Projection.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization.
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Causal mask to ensure that attention is only applied to the left in the input sequence.
        # - Flash Attention isn't (currently) compatible with ALiBi and has been removed.
        # self.register_buffer(
        #     "c_mask",
        #     torch.tril(
        #         torch.ones(
        #             config.block_size,
        #             config.block_size,
        #         )
        #     ).view(1, 1, config.block_size, config.block_size)
        # )

        # torch.tril(
        #     torch.ones(
        #         config.block_size,
        #         config.block_size
        #     )
        # ).view(1, 1, config.block_size, config.block_size)
        # """
        # tensor([[[[1., 0., 0.,  ..., 0., 0., 0.],
        #           [1., 1., 0.,  ..., 0., 0., 0.],
        #           [1., 1., 1.,  ..., 0., 0., 0.],
        #           ...,
        #           [1., 1., 1.,  ..., 1., 0., 0.],
        #           [1., 1., 1.,  ..., 1., 1., 0.],
        #           [1., 1., 1.,  ..., 1., 1., 1.]]]])
        # """

        #
        # Calculate ALiBi bias with correct shape:
        #

        # 1. The ALiBi offsets.
        #
        # e.g.
        #
        # tensor([[ 0.,  0.,  0.,  0.],
        #         [-1.,  0.,  0.,  0.],
        #         [-2., -1.,  0.,  0.],
        #      ...[-N., -2., -1.,  0.]])

        # self.register_buffer(
        #     "alibi_offset",
        #      ALiBi.bias(config.block_size)
        # )

        # 2. The ALiBi `m` values / slopes - expanded across each head.
        #
        # e.g.
        #
        # tensor([[[0.2500]],
        #         [[0.0625]],
        #         [[0.0156]],
        #         [[0.0039]]])
        #
        # Then expanded across each head: e.g.
        #
        # tensor([[[[0.2500, 0.2500, 0.2500,  ..., 0.2500, 0.2500, 0.2500],
        #           [0.2500, 0.2500, 0.2500,  ..., 0.2500, 0.2500, 0.2500],
        #           [0.2500, 0.2500, 0.2500,  ..., 0.2500, 0.2500, 0.2500],
        #           ...,
        #           [0.2500, 0.2500, 0.2500,  ..., 0.2500, 0.2500, 0.2500],
        #           [0.2500, 0.2500, 0.2500,  ..., 0.2500, 0.2500, 0.2500],
        #           [0.2500, 0.2500, 0.2500,  ..., 0.2500, 0.2500, 0.2500]],
        #
        #          [[0.0625, 0.0625, 0.0625,  ..., 0.0625, 0.0625, 0.0625],
        #           [0.0625, 0.0625, 0.0625,  ..., 0.0625, 0.0625, 0.0625],
        #           [0.0625, 0.0625, 0.0625,  ..., 0.0625, 0.0625, 0.0625],
        #           ...,
        #           [0.0625, 0.0625, 0.0625,  ..., 0.0625, 0.0625, 0.0625],
        #           [0.0625, 0.0625, 0.0625,  ..., 0.0625, 0.0625, 0.0625],
        #           [0.0625, 0.0625, 0.0625,  ..., 0.0625, 0.0625, 0.0625]],
        #
        #          [[0.0156, 0.0156, 0.0156,  ..., 0.0156, 0.0156, 0.0156],
        #           [0.0156, 0.0156, 0.0156,  ..., 0.0156, 0.0156, 0.0156],
        #           [0.0156, 0.0156, 0.0156,  ..., 0.0156, 0.0156, 0.0156],
        #           ...,
        #           [0.0156, 0.0156, 0.0156,  ..., 0.0156, 0.0156, 0.0156],
        #           [0.0156, 0.0156, 0.0156,  ..., 0.0156, 0.0156, 0.0156],
        #           [0.0156, 0.0156, 0.0156,  ..., 0.0156, 0.0156, 0.0156]],
        #
        #          [[0.0039, 0.0039, 0.0039,  ..., 0.0039, 0.0039, 0.0039],
        #           [0.0039, 0.0039, 0.0039,  ..., 0.0039, 0.0039, 0.0039],
        #           [0.0039, 0.0039, 0.0039,  ..., 0.0039, 0.0039, 0.0039],
        #           ...,
        #           [0.0039, 0.0039, 0.0039,  ..., 0.0039, 0.0039, 0.0039],
        #           [0.0039, 0.0039, 0.0039,  ..., 0.0039, 0.0039, 0.0039],
        #           [0.0039, 0.0039, 0.0039,  ..., 0.0039, 0.0039, 0.0039]]]])

        # self.register_buffer(
        #     "alibi_m",
        #     torch.tensor(
        #         ALiBi.get_slopes(config.n_head)
        #     ).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # )

    def forward(self, x):
        B, T, C = x.size()  # Batch Size, Sequence Length, Embedding Dimensionality (n_embd)
        # Note: If training data is len = 768, don't get confused between T and C (n_embd)

        q = self.c_query(x)

        if not self.use_cache or (self.cache_k is None or self.cache_v is None):
            k, v = self.c_kv(x).split(self.n_embd, dim=2)
            self.cache_k = k
            self.cache_v = v
        else:
            k, v = self.c_kv(x[:, T-1:, :]).split(self.n_embd, dim=2)
            self.cache_k = torch.cat((self.cache_k, k), dim=1)
            self.cache_v = torch.cat((self.cache_v, v), dim=1)
            k = self.cache_k
            v = self.cache_v

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)

        # Unpack the individual matrices.
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # -> (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # -> (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # -> (B, nh, T, hs)

        # Attention Is All You Need: sqrt of k-dim.
        # assert C // self.n_head == k.size(-1)
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        att = q @ k.transpose(-2, -1) / math.sqrt(C // self.n_head)

        # # Create the bias tensor for ALiBi mechanism.
        # if self.alibi_bias is None or self.alibi_bias_T != T:
        #     assert T == x.size(1)  # assuming x is of shape (Batch, Time, Features)
        #     self.alibi_bias_T = T
        #     # self.alibi_bias = (self.alibi_m * torch.ones(B, self.n_head, T, T).to(x.device)) * self.alibi_offset[:T, :T]
        #     self.alibi_bias = self.alibi_m * self.alibi_offset[:T, :T]

        # ALiBi: add the bias _after_ the query-key dot product.
        # att += self.alibi_bias

        att += (self.alibi_m.to(x.device) * self.alibi_offset[:T, :T].to(x.device))

        # Fills elements of self tensor with value where mask is True:
        #
        # Note: ALiBi bias isn't suitable as a causal mask. (Upper triangle 0s)
        # - We use `self.c_mask` instead.
        att.masked_fill_(self.c_mask[:, :, :T, :T].to(x.device) == 0, value=float('-inf'))  # Pre-computed over the whole block, select T-sized block for current sequence length.

        # Bug:
        # - Don't use `float('-inf')` (for Sep 2023 PyTorch version.)
        # - See: https://github.com/pytorch/pytorch/issues/107084
        # - Also: # https://discuss.pytorch.org/t/runtimeerror-value-cannot-be-converted-to-type-at-half-without-overflow-1e-30/109768/2

        att = F.softmax(att, dim=-1)
        # att = softmax_one(att, dim=-1)

        att = self.attn_dropout(att)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs side by side.

        y = self.c_proj(y)

        # Project values for the next in sequence or output.
        y = self.resid_dropout(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: Any):
        """
        A Multilayer Perceptron (MLP) module.

        :param config: Configuration object containing necessary hyperparameters.
        """
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, config.n_fexp * config.n_embd, bias=config.bias)  # Original is 4. A value of 2 => Regresses to noise.
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(config.n_fexp * config.n_embd, config.n_embd, bias=config.bias)  # As above.
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the MLP.

        :param x: Input tensor.
        :return: Processed tensor.
        """

        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)

        return x


class Block(nn.Module):
    def __init__(self, config: Any, c_kv: Any = None, alibi_params: Any = None):
        """
        A Block module consisting of layer normalization, attention, and MLP.

        :param config: Configuration object containing necessary hyperparameters.
        :param c_kv: Optional parameter for custom attention behavior.
        """
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalAttention(config, c_kv, alibi_params, use_cache=config.kv_cache)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the block.

        :param x: Input tensor.
        :return: Processed tensor.
        """
        x = x + self.attn(self.ln_1(x))   # Skip connection.
        x = x + self.mlp(self.ln_2(x))    # Skip connection.
        return x


@dataclass
class ParakeetConfig:
    block_size: int = 4096   # a.k.a. Context or sequence length.
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency => 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_fexp: int = 4
    dropout: float = 0.01
    bias: bool = False       # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    gqa: bool = True         # Grouped-Query Attention: Shares the `kv` tensor between N layers.
    n_blocks_per_kv: int = 2
    kv_cache: bool = True
    name: str = "parakeet4k"


class ParakeetGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = bnb.nn.StableEmbedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList(),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        #
        # Preload ALiBi requirements prior to configuring each block.
        #

        alibi_params = {
            'alibi_offset': ALiBi.bias(config.block_size),
            'alibi_m': torch.tensor(
                ALiBi.get_slopes(config.n_head)
            ).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
            'c_mask': torch.tril(
                torch.ones(
                    config.block_size,
                    config.block_size,
                )
            ).view(1, 1, config.block_size, config.block_size)
        }

        # # Create the bias tensor for ALiBi mechanism.
        # if self.alibi_bias is None or self.alibi_bias_T != T:
        #     assert T == x.size(1)  # assuming x is of shape (Batch, Time, Features)
        #     self.alibi_bias_T = T
        #     # self.alibi_bias = (self.alibi_m * torch.ones(B, self.n_head, T, T).to(x.device)) * self.alibi_offset[:T, :T]
        #     self.alibi_bias = self.alibi_m * self.alibi_offset[:T, :T]

        #
        # Matrices are loaded into the GPU once to prevent redundant memory use.
        #

        # Initialize kv matrices and blocks
        for i in range(config.n_layer):
            if i % config.n_blocks_per_kv == 0:
                c_kv = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias) if config.gqa else None
            # ParakeetGPT -> Block -> CausalAttention makes use of ALiBi params.
            block = Block(config, c_kv, alibi_params)
            self.transformer.h.append(block)

            # Unfreeze blocks by default
            for param in block.parameters():
                param.requires_grad = True

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying
        self.transformer.wte.weight = self.lm_head.weight
        # self.lm_head.weight = self.transformer.wte.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=(0.02 / math.sqrt(2 * config.n_layer)))

        print(">> Num. parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # if non_embedding:
        #     n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # Forward the model itself:
        # - These are the main steps of a decoder-only transformer.

        # 1. Tokens are embedded.
        tok_emb = self.transformer.wte(idx)    # `wte` = Vocab. Size => Embedding Size

        # 2. Optional dropout is applied.
        x = self.transformer.drop(tok_emb)

        # 3. Pass the information through the sequence of blocks.
        for block in self.transformer.h:
            x = block(x)

        # 4. Layer normalisation.
        x = self.transformer.ln_f(x)

        # 5. Train or run inference.
        if targets is not None:
            # 5a. During training, optimise for next target prediction.
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=100277)
        else:
            # 5b. During inference, convert from embedding back to vocab via linear layer.
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def configure_optimizers(self, weight_decay=0.03, learning_rate=1e-7, betas=[0.9, 0.98], device_type="cuda"):
        # Start with all the candidate parameters.
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # Filter out those that do not require grad.
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # - i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f">> Num. decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters.")
        print(f">> Num. non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters.")

        # Create AdamW optimizer and use the fused version if it is available.
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        # optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        optimizer = bnb.optim.AdamW8bit(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        # optimizer = bnb.optim.PagedLion(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        print(f">> Fused AdamW is available: {use_fused}")
        print(f">> `configure_optimizers` done, now training `{self.config.name}`...")

        return optimizer

    def generate(
            self,
            device: torch.device,
            tokenizer: BaseTokenizer,
            seed_text: str,
            max_length: int = 400,
            temperature: float = 0.70,
            freq_penalty: float = 0.2,
            pres_penalty: float = 0.2,
            top_k: int = -1,
            top_p: float = 1.00,
            min_p: float = 0.05,
            stop_sequences: list[str] = [],
            greedy: bool = False,
            token_callback: Optional[Callable[[int], None]] = None,
    ) -> str:
        """
        Generates text using a given model and tokenizer with optional temperature, frequency penalty, top-k, and greedy sampling.

        :param device: The device to run the model on.
        :param tokenizer: The tokenizer corresponding to the model.
        :param seed_text: The initial text to start the generation.
        :param max_length: The maximum length of the generated text (default is 400).
        :param temperature: The temperature for controlling randomness in sampling (default is 0.7).
        :param freq_penalty: The frequency penalty for controlling token repetition (default is 0.02).
        :param pres_penalty: The presence penalty for controlling token existence (default is 0.02).
        :param top_k: The number of top tokens considered for sampling (default is 70).
        :param top_p: <Insert intuitive explanation here>
        :param min_p: Probability cut-off for logits.
        :param stop_sequences: A list of strings that will end token generation if encountered.
        :param greedy: Just select the logit with the highest probability at each step.
        :param token_callback: A function that will be called for each generated token, receiving the token ID as input.
        :return: The generated text.
        """

        # Set model to evaluation mode (disables features like dropout during inference).
        self.eval()

        # Clear KV cache of the model. (TODO: Move into the model itself)
        for block in range(len(self.transformer.h)):
            self.transformer.h[block].attn.cache_k = None
            self.transformer.h[block].attn.cache_v = None

        # Check if temperature is very low, and if so, enable greedy sampling.
        if temperature < 0.01:
            greedy = True

        # Tokenize seed_text into input_ids.
        input_ids = tokenizer.encode(seed_text)

        # Initialize token_count dictionary to keep track of token frequencies.
        token_count: defaultdict = defaultdict(int)
        for token in input_ids:
            token_count[token] += 1

        # Convert `input_ids` to tensor, add batch dimension then send to device.
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

        # Disable gradient computation for inference.
        with torch.no_grad():
            # Iterate until maximum length is reached.
            for _ in range(max_length):
                # Forward pass through model to get logits (sparse vocab-sized matrix).
                logits, _ = self(input_ids)

                # Apply frequency and presence penalties directly to logits:
                #
                # GPT: mu[j] -> mu[j] - c[j] * alpha_frequency - float(c[j] > 0) * alpha_presence
                #
                # Where:

                # mu[j] is the logits of the j-th token
                # c[j] is how often that token was sampled prior to the current position
                # float(c[j] > 0) is 1 if c[j] > 0 and 0 otherwise
                # alpha_frequency is the frequency penalty coefficient
                # alpha_presence is the presence penalty coefficient
                # As we can see, the presence penalty is a one-off additive contribution that applies to all tokens that have been sampled at least once
                # and the frequency penalty is a contribution that is proportional to how often a particular token has already been sampled.

                # Reasonable values for the penalty coefficients are around 0.1 to 1 if the aim is to just reduce repetitive samples somewhat.
                # If the aim is to strongly suppress repetition, then one can increase the coefficients up to 2, but this can noticeably degrade the quality of samples.
                # Negative values can be used to increase the likelihood of repetition.

                # Apply frequency and presence penalties directly to logits
                for token_id, count in token_count.items():
                    logits[0, -1, token_id] -= count * freq_penalty        # Freq. penalty
                    logits[0, -1, token_id] -= (count > 0) * pres_penalty  # Pres. penalty

                if greedy:
                    # Select token with highest logit and add batch dimension.
                    next_token = torch.argmax(logits[..., -1, :], dim=-1).unsqueeze(0)
                else:
                    if top_k is not None:
                        # Retrieve top_k logits and their indices.
                        top_k_values, top_k_indices = torch.topk(logits, k=top_k, dim=-1)
                        # Create tensor with top_k logits only.
                        logits = torch.zeros_like(logits).scatter(-1, top_k_indices, top_k_values)

                    # Compute probabilities using temperature scaling:
                    # - A high temp makes the outcomes more even (less confident), while a low temp makes certain outcomes stand out more (more confident).

                    # 2024-07-13: `...logits[..., -1, :] / temperature` is "Temperature First".
                    # - See: https://www.reddit.com/r/LocalLLaMA/comments/17vonjo/your_settings_are_probably_hurting_your_model_why/
                    probs = torch.nn.functional.softmax(logits[..., -1, :], dim=-1).unsqueeze(0)

                    # Get shape of probabilities tensor.
                    b, t, c = probs.shape

                    # Reshape probabilities.
                    probs = probs.reshape(b * t, c)

                    # Sort probabilities and apply top-p (nucleus) sampling:
                    # - Indices correlate with token index from the sparse vocab tensor.
                    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)

                    if top_p is not None:
                        # `top_p` sampling:
                        # - Accumulate values left to right and return the cumulative array as a result. [0, 1, 2, 3] => [0, 1, 3, 6]
                        probs_sum = torch.cumsum(probs_sort, dim=-1)

                        # cumulative_mask = cumulative_probs <= top_p
                        # probs_mask = probs_sum <= top_p
                        probs_mask = probs_sum - probs_sort > top_p
                        probs_sort[probs_mask] = 0.0

                    # `min_p` filtering.
                    if min_p is not None:
                        _min_p = probs_sort.max() * min_p
                        min_p_mask = probs_sort >= _min_p
                        if min_p_mask.sum() > 0:
                            probs_sort = probs_sort[:, min_p_mask[0]]
                            probs_idx = probs_idx[:, min_p_mask[0]]

                    # Post-sampled temperature scaling...
                    probs_sort /= temperature

                    # Re-distribute over 1. aka. normalize the truncated distribution.
                    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

                    # `torch.multinomial` will sample N values and return their indices (in this case just 1).
                    next_token = torch.multinomial(probs_sort, num_samples=1)
                    next_token = torch.gather(probs_idx, -1, next_token)

                    # Increment token count for the sampled token.
                    if next_token.item():
                        token_count[next_token.item()] = token_count.get(next_token.item(), 0) + 1

                # Disallowed tokens: <|padding|>
                if next_token.item() == tokenizer.pad_token:
                    break

                # Check if the generated text ends with any of the stop sequences
                generated_text = tokenizer.decode(input_ids[0].tolist())
                if any([(stop_sequence in generated_text) for stop_sequence in stop_sequences]):
                    break

                # Break if end-of-text token is reached
                # Changed: Checking for end-of-text token based on token id, not tensor!
                if next_token.item() == tokenizer.eop_token:
                    break

                # Concatenate newly predicted token to the input sequence.
                input_ids = torch.cat((input_ids, next_token), dim=-1)

                if token_callback is not None:
                    token_callback(tokenizer.decode([next_token.item()]))

        # Decode the tokenized input_ids back to text
        try:
            output_text = tokenizer.decode(input_ids[0].tolist())
        except Exception as e:  # Handle any exceptions during decoding
            output_text = f"Failed to decode model output: {e}..."
            print(output_text)

        self.train()  # Set model back to training mode
        return output_text[len(seed_text):]  # Return the generated text
