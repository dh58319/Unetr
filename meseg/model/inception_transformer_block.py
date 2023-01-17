import torch
import torch.nn as nn
import numpy as np


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self, 
        dim : int, 
        num_heads : int,
        qkv_bias : bool = False,
        attn_drop : float = 0.,
        proj_drop : float = 0.
    ) -> None:
        super(MultiHeadSelfAttention, self).__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        # assert D % self.num_heads == 0, "D of x should be divisible by num_heads"
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, D//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # each shape = (B, self.num_heads, N, D//self.num_heads)

        attn = (q @ k.transpose(-2,-1)) * self.scale
        attn_probs = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_probs)

        x = (attn @ v).transpose(1,2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_probs



class Mlp(nn.Module):
    def __init__(
        self,
        in_features : int,
        hidden_features : int,
        out_features : int,
        act_layer = nn.GELU,
        bias : bool = True,
        drop : float = 0.,
    ) -> None:
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(x))
        return x



class Block(nn.Module):
    r"""
    """
    def __init__(
        self,
        dim : int,
        num_heads : int,
        mlp_ratio : int = 4,
        qkv_bias : bool = False,
        attn_drop : float = 0.,
        proj_drop : float = 0.,
        act_layer = nn.GELU,
        norm_layer = nn.LayerNorm,
    ) -> None:
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads, qkv_bias, attn_drop, proj_drop
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim*mlp_ratio), dim, act_layer)
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        skip = x
        x, attn_probs = self.attn(self.norm1(x))
        x = skip + x
        x = x + self.mlp(self.norm2(x))
        return x, attn_probs



class InceptionBlock(nn.Module):
    r"""
    num_path: a factor that determines how many inception paths are created.
        so, num_path 1 means gerneral attention.
    """
    def __init__(
        self,
        dim : int,
        num_heads : int,
        mlp_ratio : float = 4.,
        qkv_bias : bool = False,
        num_path : int = 1,
        pixshuf_factor : int = 1,
        attn_drop : float = 0.,
        proj_drop : float = 0.,
        act_layer = nn.GELU,
        norm_layer = nn.LayerNorm,
        concat : bool = True,
    ) -> None:
        super(InceptionBlock, self).__init__()
        assert num_path % 2 == 1, f"num_path must be odd number"

        self.dim = dim
        self.num_path = num_path
        self.median = (num_path // 2) + 1
        self.total_dim = dim * self.num_path
        for i in range(1, self.num_path+1):
            if i < self.median:
                new_dim = dim * (pixshuf_factor**(i*2))
                self.add_module(
                    f"pixunshuf{i}",
                    nn.PixelUnshuffle(pixshuf_factor*i)
                )
            elif i > self.median:
                new_dim = int(dim / (pixshuf_factor**((i-self.median)*2)))
                self.add_module(
                    f"pixshuf{i}",
                    nn.PixelShuffle(pixshuf_factor*(i-self.median))
                )
            else:   new_dim = dim
            self.add_module(
                f"path{i}", 
                Block(
                    dim=new_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    act_layer=act_layer,
                    norm_layer=norm_layer
                )
            )
            # self.total_dim = self.total_dim + new_dim
        
        self.concat = concat
        if self.concat:
            self.norm = norm_layer(self.total_dim)
            self.fc = nn.Linear(self.total_dim, dim)
        
    

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        H = W = int(np.sqrt(N))

        x = x.permute(0,2,1).view(B,D,H,W)

        features = []
        for i in range(1, self.num_path+1):
            if i < self.median:
                x = getattr(self, f"pixunshuf{i}")(x)
            elif i > self.median:
                x = getattr(self, f"pixshuf{i}")(x)
            else:   pass

            B, D, H, W = x.shape
            x = x.contiguous().view(B,D,H*W).permute(0,2,1)
            x, attn_probs = getattr(self, f"path{i}")(x)
            x = x.permute(0,2,1).view(B,D,H,W)

            if i < self.median:
                x = getattr(self, f"pixshuf{self.num_path+1-i}")(x)
            elif i > self.median:
                x = getattr(self, f"pixunshuf{self.num_path+1-i}")(x)
            else:   pass
            features.append(x)
        

        # for f in features:
        #     print("features shape: ", f.shape)
        if self.concat:
            x = torch.cat(features,dim=1)
            x = x.view(B,self.total_dim,N).permute(0,2,1)
            x = self.fc(self.norm(x))
        else:
            x = features[0]
            for f in features[1:]:
                x = x + f
            x = x.view(B,self.dim,N).permute(0,2,1)
        return x, attn_probs