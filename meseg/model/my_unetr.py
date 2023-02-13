import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from monai.networks.nets import SwinUNETR


class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class DLA(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, expand_ratio=3, refine_mode='conv'):
        super(DLA, self).__init__()
        """
            Distributed Local Attention used for refining the attention map.
        """

        hidden_dim = round(inp * expand_ratio)
        self.expand_ratio = expand_ratio
        self.identity = stride == 1 and inp == oup
        self.inp, self.oup = inp, oup
        self.high_dim_id = False
        self.refine_mode = refine_mode

        if refine_mode == 'conv':
            self.conv = Conv2dSamePadding(hidden_dim, hidden_dim, (kernel_size, kernel_size), stride, (1, 1), groups=1,
                                          bias=False)
        elif refine_mode == 'conv_expand':
            if self.expand_ratio != 1:
                self.conv_exp = Conv2dSamePadding(inp, hidden_dim, 1, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.depth_sep_conv = Conv2dSamePadding(hidden_dim, hidden_dim, (kernel_size, kernel_size), stride, (1, 1),
                                                    groups=hidden_dim, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_dim)

            self.conv_pro = Conv2dSamePadding(hidden_dim, oup, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(oup)

            self.relu = nn.ReLU6(inplace=True)

    def forward(self, input):
        x = input
        if self.refine_mode == 'conv':
            return self.conv(x)
        else:
            if self.expand_ratio != 1:
                x = self.relu(self.bn1(self.conv_exp(x)))
            x = self.relu(self.bn2(self.depth_sep_conv(x)))
            x = self.bn3(self.conv_pro(x))
            if self.identity:
                return x + input
            else:
                return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class PatchEmbed(nn.Module):
    def __init__(self,input_dim,embed_dim,patch_size):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.BatchNorm3d(embed_dim)
        self.se = SELayer(embed_dim)
        self.conv2  = nn.Conv3d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        v = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.se(x)

        return x


class SingleDeconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.block = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=2, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.block(x)


class SingleConv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=1,
                               padding=((kernel_size - 1) // 2))

    def forward(self, x):
        return self.block(x)


class Conv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleConv3DBlock(in_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Deconv3DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super().__init__()
        self.block = nn.Sequential(
            SingleDeconv3DBlock(in_planes, out_planes),
            SingleConv3DBlock(out_planes, out_planes, kernel_size),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout, mul_head=1):
        super().__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(embed_dim, self.all_head_size)
        self.key = nn.Linear(embed_dim, self.all_head_size)
        self.value = nn.Linear(embed_dim, self.all_head_size)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1)

        self.vis = False
        
        #########exp- donghyun ############
        self.mul_head = mul_head
        if self.mul_head > 1:
            self.conv_attn = nn.Sequential(
                nn.Conv2d(in_channels=num_heads,out_channels=num_heads*mul_head, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(num_features=num_heads*mul_head),
                nn.Conv2d(in_channels=num_heads*mul_head ,out_channels=num_heads, kernel_size=1),
            )

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if self.mul_head > 1:
            attention_scores = self.conv_attn(attention_scores)
            
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, in_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1()
        x = self.act(x)
        x = self.drop(x)
        return x


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=786, d_ff=2048, dropout=0.1):
        super().__init__()
        # Torch linears have a `b` by default.
        self.w_1 = nn.Linear(d_model, d_ff)#2,216,2048
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        print("w1",x.shape)
        #x = self.conv(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.w_2(x)
        print("w2",x.shape)
        return x


class Embeddings(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, dropout):
        super().__init__()
        self.n_patches = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # self.patch_embeddings = nn.Conv3d(in_channels=input_dim, out_channels=embed_dim,
        #                                   kernel_size=patch_size, stride=patch_size)
        self.patch_embeddings = PatchEmbed(input_dim, embed_dim, patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = nn.Sequential(
            #conv3d 2,2048,6,6,6
            nn.Conv3d(in_features, hidden_features, 1, 1, 0, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features, eps=1e-5),
        )
        self.proj = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.proj_act = nn.GELU()
        self.proj_bn = nn.BatchNorm2d(hidden_features, eps=1e-5)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True),
            nn.BatchNorm2d(out_features, eps=1e-5),
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        B, N, C = x.shape
        H,W,R = 6,6,6
        x = x.permute(0, 2, 1).reshape(2,768,6,6,6)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.proj(x) + x
        x = self.proj_act(x)
        x = self.proj_bn(x)
        x = self.conv2(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.drop(x)
        return x
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, cube_size, patch_size, mul_head=1):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.mlp_dim = int((cube_size[0] * cube_size[1] * cube_size[2]) / (patch_size * patch_size * patch_size))
        #self.mlp = PositionwiseFeedForward(embed_dim, 2048)
        self.mlp = Mlp(embed_dim, hidden_features=2048, out_features=embed_dim)
        self.attn = SelfAttention(num_heads, embed_dim, dropout, mul_head= mul_head)
        self.attn_refine = Refined_Attention(dim= embed_dim,num_heads=num_heads,attn_drop= dropout, refine_mode='conv')
    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        #x,weights= self.attn_refine(x)
        x = x + h
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)

        x = x + h
        return x, weights


class Transformer(nn.Module):
    def __init__(self, input_dim, embed_dim, cube_size, patch_size, num_heads, num_layers, dropout, extract_layers, mul_head=1):
        super().__init__()
        self.embeddings = Embeddings(input_dim, embed_dim, cube_size, patch_size, dropout)
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.extract_layers = extract_layers
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, num_heads, dropout, cube_size, patch_size, mul_head = mul_head)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        extract_layers = []
        hidden_states = self.embeddings(x)

        for depth, layer_block in enumerate(self.layer):
            hidden_states, _ = layer_block(hidden_states)
            if depth + 1 in self.extract_layers:
                extract_layers.append(hidden_states)

        return extract_layers


class Refined_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., expansion_ratio=3,
                 share_atten=False, apply_transform=True, refine_mode='conv', kernel_size=3, head_expand=3):
        """
            refine_mode: "conv" represents only convolution is used for refining the attention map;
                         "conv-expand" represents expansion and conv are used together for refining the attention map;
            share_atten: If set True, the attention map is not generated; use the attention map generated from the previous block
        """
        super().__init__()
        self.num_heads = num_heads
        self.share_atten = share_atten
        head_dim = dim // num_heads
        self.apply_transform = apply_transform

        self.scale = qk_scale or head_dim ** -0.5

        if self.share_atten:
            self.DLA = DLA(self.num_heads, self.num_heads, refine_mode=refine_mode)
            self.adapt_bn = nn.BatchNorm2d(self.num_heads)

            self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        elif apply_transform:
            self.DLA = DLA(self.num_heads, self.num_heads, kernel_size=kernel_size, refine_mode=refine_mode,
                           expand_ratio=head_expand)
            self.adapt_bn = nn.BatchNorm2d(self.num_heads)
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, atten=None):
        B, N, C = x.shape
        if self.share_atten:
            attn = atten
            attn = self.adapt_bn(self.DLA(attn)) * self.scale

            v = self.qkv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            attn_next = atten
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            q = math.sqrt(self.scale) * q
            k = math.sqrt(self.scale) * k
            attn = (q @ k.transpose(-2, -1))  # * self.scale
            attn = attn.softmax(dim=-1) + atten * self.scale if atten is not None else attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            if self.apply_transform:
                attn = self.adapt_bn(self.DLA(attn))
            attn_next = attn
        x = (attn @ v).transpose(1, 2).reshape(B, attn.shape[-1], C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_next


class UNETR2(nn.Module):
    def __init__(self, img_shape=(96, 96, 96), input_dim=4, output_dim=3, embed_dim=768, patch_size=16, num_heads=12, dropout=0.1,mul_head=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.img_shape = img_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.num_layers = 12
        self.ext_layers = [3, 6, 9, 12]

        self.patch_dim = [int(x / patch_size) for x in img_shape]



        # Transformer Encoder
        self.transformer = \
            Transformer(
                input_dim,
                embed_dim,
                img_shape,
                patch_size,
                num_heads,
                self.num_layers,
                dropout,
                self.ext_layers,
                mul_head=mul_head
            )

        # U-Net Decoder
        self.decoder0 = \
            nn.Sequential(
                Conv3DBlock(input_dim, 32, 3),
                Conv3DBlock(32, 64, 3),
                SELayer(64)
            )

        self.decoder3 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
                Deconv3DBlock(256, 128)
            )

        self.decoder6 = \
            nn.Sequential(
                Deconv3DBlock(embed_dim, 512),
                Deconv3DBlock(512, 256),
            )

        self.decoder9 = \
            Deconv3DBlock(embed_dim, 512)

        self.decoder12_upsampler = \
            SingleDeconv3DBlock(embed_dim, 512)

        self.decoder9_upsampler = \
            nn.Sequential(
                Conv3DBlock(1024, 512),
                Conv3DBlock(512, 512),
                Conv3DBlock(512, 512),
                SingleDeconv3DBlock(512, 256)
            )

        self.decoder6_upsampler = \
            nn.Sequential(
                Conv3DBlock(512, 256),
                Conv3DBlock(256, 256),
                SELayer(256),
                SingleDeconv3DBlock(256, 128)
            )

        self.decoder3_upsampler = \
            nn.Sequential(
                Conv3DBlock(256, 128),
                Conv3DBlock(128, 128),
                SingleDeconv3DBlock(128, 64)
            )

        self.decoder0_header = \
            nn.Sequential(
                Conv3DBlock(128, 64),
                Conv3DBlock(64, 64),
                SELayer(64),
                SingleConv3DBlock(64, output_dim, 1)
            )

    def forward(self, x):
        z = self.transformer(x)
        z0, z3, z6, z9, z12 = x, *z
        z3 = z3.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        print("z3shape=",z3.shape)
        z6 = z6.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        print("z6shape=",z6.shape)
        z9 = z9.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        print("z9shape=",z9.shape)
        z12 = z12.transpose(-1, -2).view(-1, self.embed_dim, *self.patch_dim)
        print("z12shape=",z12.shape)

        z12 = self.decoder12_upsampler(z12)
        z9 = self.decoder9(z9)
        z9 = self.decoder9_upsampler(torch.cat([z9, z12], dim=1))
        z6 = self.decoder6(z6)
        z6 = self.decoder6_upsampler(torch.cat([z6, z9], dim=1))
        z3 = self.decoder3(z3)
        z3 = self.decoder3_upsampler(torch.cat([z3, z6], dim=1))
        z0 = self.decoder0(z0)
        output = self.decoder0_header(torch.cat([z0, z3], dim=1))
        return output
    
    
if __name__ == '__main__':
    x = torch.rand(2,1,96,96,96)
    print('practice unetr')
    print(x.shape)
    model = UNETR2((96,96,96),1,14,mul_head=3)
    y = model(x)
    print(f"input of model shape:{x.shape}") 
    print(f"output of model shape:{y.shape}")