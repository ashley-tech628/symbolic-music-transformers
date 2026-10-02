"""Model architectures extracted from the selected research notebook."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class MetricPositionEmbedding(nn.Module):
    """Learned embedding for position within the musical bar (beat awareness).

    A 4/4 bar at eighth-note resolution has 8 positions:
      0 = downbeat, 2 = beat 2, 4 = beat 3, 6 = beat 4  (strong beats)
      1, 3, 5, 7                                         (off-beats)
    Telling the model where it is within the bar is a domain-specific
    inductive bias: cadences tend to fall on strong beats, passing notes
    on off-beats.  This replaces / augments the plain absolute position
    embedding used in vanilla Transformers.
    """
    def __init__(self, d, steps_per_bar=8):
        super().__init__()
        self.spb = steps_per_bar
        self.emb = nn.Embedding(steps_per_bar, d)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, T, device):
        pos = torch.arange(T, device=device) % self.spb
        return self.emb(pos).unsqueeze(0)

class RelMultiheadSelfAttn(nn.Module):
    """Multi-head self-attention with a learned relative-position bias (Shaw et al. 2018)."""
    def __init__(self, d, h, max_rel, drop):
        super().__init__()
        assert d % h == 0
        self.h, self.dk, self.max_rel = h, d//h, max_rel
        self.qkv = nn.Linear(d, 3*d); self.out = nn.Linear(d, d); self.drop = nn.Dropout(drop)
        self.rel = nn.Parameter(torch.zeros(2*max_rel+1, h)); nn.init.normal_(self.rel, std=0.02)
    def forward(self, x, causal=True, key_padding_mask=None):
        B,T,D = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B,T,self.h,self.dk).transpose(1,2)
        k = k.view(B,T,self.h,self.dk).transpose(1,2)
        v = v.view(B,T,self.h,self.dk).transpose(1,2)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.dk)           # B,h,T,T
        pos = torch.arange(T, device=x.device)
        rel = (pos[None,:]-pos[:,None]).clamp(-self.max_rel,self.max_rel)+self.max_rel
        scores = scores + self.rel[rel].permute(2,0,1).unsqueeze(0)      # + relative bias
        if causal:
            scores = scores.masked_fill(torch.triu(torch.ones(T,T,device=x.device,dtype=torch.bool),1), float('-inf'))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:,None,None,:], float('-inf'))
        a = self.drop(F.softmax(scores, dim=-1))
        o = (a @ v).transpose(1,2).contiguous().view(B,T,D)
        return self.out(o)

class RelDecoderBlock(nn.Module):
    def __init__(self, d, h, ff, max_rel, drop):
        super().__init__()
        self.ln1=nn.LayerNorm(d); self.attn=RelMultiheadSelfAttn(d,h,max_rel,drop)
        self.ln2=nn.LayerNorm(d)
        self.mlp=nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Dropout(drop),nn.Linear(ff,d))
        self.drop=nn.Dropout(drop)
    def forward(self, x, kpm=None):
        x = x + self.drop(self.attn(self.ln1(x), causal=True, key_padding_mask=kpm))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

class RelTransformerLM(nn.Module):
    """Decoder-only LM with:
      - relative-position self-attention (Shaw et al. 2018 / Music Transformer)
      - metric position embedding (beat awareness)
      - weight tying between embedding and output projection
    """
    def __init__(self, V, d=256, h=8, layers=5, ff=1024, max_rel=128, drop=0.1, beat_steps=8):
        super().__init__()
        self.emb      = nn.Embedding(V, d)
        self.beat_emb = MetricPositionEmbedding(d, beat_steps)
        self.blocks   = nn.ModuleList([RelDecoderBlock(d, h, ff, max_rel, drop) for _ in range(layers)])
        self.ln       = nn.LayerNorm(d)
        self.head     = nn.Linear(d, V)
        self.head.weight = self.emb.weight          # weight tying
    def forward(self, x, kpm=None):
        T   = x.size(1)
        hdn = self.emb(x) + self.beat_emb(T, x.device)   # token + beat position
        for b in self.blocks: hdn = b(hdn, kpm)
        return self.head(self.ln(hdn))

class Seq2SeqHarmonizer(nn.Module):
    """Bidirectional melody encoder + autoregressive harmony decoder with cross-attention."""
    def __init__(self, mV, hV, d=256, heads=8, e_layers=3, d_layers=3, ff=1024, drop=0.1, max_len=512):
        super().__init__()
        self.mel_emb=nn.Embedding(mV,d); self.harm_emb=nn.Embedding(hV,d); self.pos=nn.Embedding(max_len,d)
        enc=nn.TransformerEncoderLayer(d,heads,ff,drop,activation='gelu',batch_first=True)
        self.encoder=nn.TransformerEncoder(enc,e_layers)
        dec=nn.TransformerDecoderLayer(d,heads,ff,drop,activation='gelu',batch_first=True)
        self.decoder=nn.TransformerDecoder(dec,d_layers)
        self.ln=nn.LayerNorm(d); self.head=nn.Linear(d,hV)
    def encode(self, mel, mel_pad):
        T=mel.size(1); p=torch.arange(T,device=mel.device).unsqueeze(0)
        return self.encoder(self.mel_emb(mel)+self.pos(p), src_key_padding_mask=mel_pad)   # bidirectional
    def decode(self, din, memory, mem_pad, din_pad):
        T=din.size(1); p=torch.arange(T,device=din.device).unsqueeze(0)
        m=torch.triu(torch.ones(T,T,device=din.device,dtype=torch.bool),1)                 # causal
        h=self.decoder(self.harm_emb(din)+self.pos(p), memory, tgt_mask=m,
                       tgt_key_padding_mask=din_pad, memory_key_padding_mask=mem_pad)        # cross-attn
        return self.head(self.ln(h))
    def forward(self, mel, din, mel_pad=None, din_pad=None):
        return self.decode(din, self.encode(mel, mel_pad), mel_pad, din_pad)
