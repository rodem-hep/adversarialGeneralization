# Transformer model for point cloud data (-Matthew Leigh and Franck Rothen)

import torch as torch
from torch import nn

from torch.nn.functional import scaled_dot_product_attention

class SimpleTransformer(nn.Module):
    def __init__(self,
                output_dim: int = 1,  
                input_dim: int = 4, 
                dim: int = 64, 
                num_heads: int = 4,
                num_layers: int = 3,
                ff_mult: int = 2,
                dropout: float = 0.1,
                do_final_norm: bool = True,
    ) -> None:
        
        super().__init__()

        # Define the transformer
        self.transformer = ConditionalSequential(
            Encoder(
                inpt_dim = input_dim,
                dim = dim,
                num_layers = num_layers,
                ff_mult = ff_mult,
                dropout = dropout,
                num_heads = num_heads,
                do_final_norm = do_final_norm,
            ),
            ClassAttention(
                output_dim = output_dim,
                dim = dim,
                ff_mult = ff_mult,
                num_heads = num_heads,
                dropout = dropout,
            )
        )
    
    def forward(self, x: torch.Tensor, kv_mask: torch.BoolTensor | None = None, ctxt: torch.Tensor | None = None) -> torch.Tensor:
        return self.transformer(x, kv_mask=kv_mask, ctxt=ctxt)

        
def attach_context(x: torch.Tensor, ctxt: torch.Tensor | None = None) -> torch.Tensor:
    """Concat a tensor with context which has the same or lower dimensions.

    New dimensions are added at index 1
    """
    if ctxt is None:
        return x
    if (dim_diff := x.dim() - ctxt.dim()) > 0:
        ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
        ctxt = ctxt.expand(*x.shape[:-1], -1)
    return torch.cat((x, ctxt), dim=-1)

class MHAttention(nn.Module):
    def __init__(
        self,
        num_heads: int = 8,
        dim: int = 128,
        ctxt_dim: int = 0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ctxt_dim = ctxt_dim
        self.lin_q =  nn.Linear(dim, dim)
        self.lin_k = nn.Linear(dim, dim)
        self.lin_v = nn.Linear(dim, dim)
        self.lin_o = nn.Linear(dim, dim)
        if ctxt_dim:
            self.lin_c = nn.Linear(dim+ctxt_dim, dim)

    def forward(
            self,
            x: torch.Tensor,
            kv: torch.Tensor | None = None,
            kv_mask: torch.BoolTensor | None = None,
            ctxt: torch.Tensor | None = None,
        ):

        # If context is expected, mix it in
        if self.ctxt_dim:
            x = attach_context(x, ctxt)
            x = self.lin_c(x)

        # If kv is none, then we are doing self attention
        if kv is None:
            kv = x

        # Projections
        q = self.lin_q(x)
        k = self.lin_k(kv)
        v = self.lin_v(kv)

        # Split the final dimension from dim into (num_heads, head_dim)
        # Shape is (batch, num_nodes, num_heads, head_dim)
        q = q.view(x.size(0), -1, self.num_heads, self.head_dim)
        k = k.view(x.size(0), -1, self.num_heads, self.head_dim)
        v = v.view(x.size(0), -1, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, num_nodes, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate the attention bias such the no-padded element is not considered
        if kv_mask is not None:
            attn_bias = kv_mask.unsqueeze(-2).expand(-1, q.shape[-2], -1)
            attn_bias = attn_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        else:
            attn_bias = 0

        # Attention (much faster - replaces the commented out steps below)
        attn_out = scaled_dot_product_attention(q, k, v, attn_bias)
        # attn_bias = torch.where(attn_bias, 0, -torch.inf)
        # attn_weight = q @ k.transpose(-2, -1) / math.sqrt(self.dim)
        # attn_weight = torch.nn.functional.softmax(attn_weight + attn_bias, dim=-1)
        # attn_out = attn_weight @ v

        # Merge the final two dimensions back together
        # Shape is (batch, num_nodes, dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(-1, x.size(1), self.dim)

        # Final mixing layer
        return self.lin_o(attn_out)

class SwliGLU(nn.Module):
    def __init__(self, dim: int = 128, ff_mult: int = 4, dropout: float = 0.0, ctxt_dim: int = 0):
        super().__init__()
        self.dim = dim
        self.ctxt_dim = ctxt_dim
        self.lin1 = nn.Linear(dim+ctxt_dim, dim * ff_mult * 2)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(dim * ff_mult, dim)

    def forward(self, x, ctxt: torch.Tensor | None = None) -> torch.Tensor:
        if self.ctxt_dim:
            x = attach_context(x, ctxt)
        x1, x2 = self.lin1(x).chunk(2, dim=-1)
        return self.lin2(self.dropout(self.act(x1)) * x2)

class PreNormResidual(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.dim = dim
        self.ln = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return x + self.fn(self.ln(x), **kwargs)

class EncoderBlock(nn.Module):
    """MHSA + FF"""
    def __init__(
            self,
            dim: int = 128,
            ctxt_dim: int = 0,
            ff_mult: int = 4,
            num_heads: int = 8,
            dropout: float = 0.0
    ):
        super().__init__()
        self.attn = PreNormResidual(dim, MHAttention(num_heads, dim, ctxt_dim))
        self.ff = PreNormResidual(dim, SwliGLU(dim, ff_mult, dropout, ctxt_dim))

    def forward(
            self,
            x: torch.Tensor,
            kv: torch.Tensor | None = None,
            kv_mask: torch.BoolTensor | None = None,
            ctxt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.attn(x, kv=kv, kv_mask=kv_mask, ctxt=ctxt)
        return self.ff(x, ctxt=ctxt)

class ClassAttention(nn.Module):
    def __init__(
        self,
        output_dim: int,
        dim: int = 128,
        ctxt_dim: int = 0,
        ff_mult: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        self.attn = MHAttention(num_heads, dim, ctxt_dim=ctxt_dim)
        self.ff = SwliGLU(dim, ff_mult, dropout, ctxt_dim=ctxt_dim)
        self.output_proj = nn.Linear(dim, output_dim)
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x: torch.Tensor, kv_mask: torch.BoolTensor | None = None, ctxt: torch.Tensor | None = None) -> torch.Tensor:
        # Expand the class token across the batch
        class_token = self.class_token.expand(x.size(0), -1, -1)
        class_token = self.attn(class_token, kv=x, kv_mask=kv_mask, ctxt=ctxt)
        class_token = self.ff(class_token, ctxt=ctxt)
        return self.output_proj(class_token).squeeze(1)


class Encoder(nn.Module):
    def __init__(
        self,
        inpt_dim: int,
        dim: int = 128,
        ctxt_dim: int = 0,
        num_layers: int = 6,
        ff_mult: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        do_final_norm: bool = True,
    ):
        super().__init__()
        self.inpt_proj = nn.Linear(inpt_dim, dim)
        self.layers = nn.ModuleList([
            EncoderBlock(dim, ctxt_dim, ff_mult, num_heads, dropout) for _ in range(num_layers)
        ])
        if do_final_norm:
            self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, kv_mask: torch.BoolTensor | None = None, ctxt: torch.Tensor | None = None) -> torch.Tensor:
        x = self.inpt_proj(x)
        for layer in self.layers:
            x = layer(x, kv_mask=kv_mask, ctxt=ctxt)
        if hasattr(self, "final_norm"):
            x = self.final_norm(x)
        return x

class ConditionalSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


def main():

    batch_size = 32
    num_nodes = 10
    num_features = 4
    ctxt_features = 2

    pc_to_vect = ConditionalSequential(
        Encoder(
            inpt_dim=num_features,
            ctxt_dim=ctxt_features,
            dim = 64,
            num_layers = 6,
            ff_mult = 2,
            num_heads = 4,
        ),
        ClassAttention(
            output_dim=10,
            ctxt_dim=ctxt_features,
            dim = 64,
            ff_mult = 1,
            num_heads = 2,
        )
    )

    x = torch.randn(batch_size, num_nodes, num_features)
    ctxt = torch.randn(batch_size, ctxt_features)
    mask = torch.randn(batch_size, num_nodes) > 0
    mask[:, 0] = True

    output = pc_to_vect(x, kv_mask=mask, ctxt=ctxt)

if __name__ == "__main__":
    main()