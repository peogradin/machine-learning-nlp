# %%
import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig
from torch.distributions.categorical import Categorical

class A2ModelConfig(PretrainedConfig):
    """Configuration object that stores hyperparameters that define the Transformer language model."""
    def __init__(self, vocab_size=None, hidden_size=None, intermediate_size=None, num_attention_heads=None, 
                 num_hidden_layers=None,
                 rope_theta=None, hidden_act='silu', max_position_embeddings=None, rms_norm_eps=None, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.num_attention_heads = num_attention_heads
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers

class A2MLP(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')
        # TODO: initalize components here
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.hidden_act = nn.functional.silu

    def forward(self, hidden_states):
        return self.down_proj(self.hidden_act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

# %% Sanity check
hidden_size = 10
intermediate_size = 15
config = A2ModelConfig(None, hidden_size=hidden_size, intermediate_size=intermediate_size)
mlp = A2MLP(config)

test_tensor = torch.ones(hidden_size)
out = mlp(test_tensor)
print(out.shape, out)
# %%

class A2MLP2(nn.Module):
    """The MLP layer of the Transformer. Uses the SwiGLU architecture."""
    def __init__(self, config):
        super().__init__()
        assert(config.hidden_act == 'silu')

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states_1, hidden_states_2 = hidden_states.chunk(2, dim=-1)
        # SwiLU activation on one half and element-wise multiply with the other half
        hidden_states = self.silu(hidden_states_1) * hidden_states_2
        # Final linear layer
        hidden_states = self.fc2(hidden_states)
        return hidden_states
    
mlp = A2MLP(A2ModelConfig(
    vocab_size=1000,
    hidden_size=100,
    intermediate_size=200,
))
# some 3-dimensional tensor where the last dimension has the same size as hidden_size
x = torch.randn(2, 10, 100)
output = mlp(x)
print(output.shape)  # should be the same shape (2, 10, 100)

# This is optional, since you can use PyTorch's RMSNorm.
class A2RMSNorm(nn.Module):
    """RMS layer normalization."""
    def __init__(self, config):
        super().__init__()
        # TODO: Use config.rms_norm_eps
        # TODO: initalize weights here

    def forward(self, hidden_states):
        ...


class A2Attention(nn.Module):
    """The multi-head attention layer of the Transformer. Uses standard scaled dot-product attention with causal masking."""
    
    def __init__(self, config):
        super().__init__()
        # TODO: set up W_q, W_k, W_v, W_o here
        # TODO: set up normalizers here
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_h = config.num_attention_heads
        assert(self.hidden_size % self.n_h == 0)
        self.d_h = self.hidden_size // self.n_h
        self.W_Q = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_K = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_O = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.config.rms_norm_eps)

    def forward(self, hidden_states, rope_rotations):
        b, m, d  = hidden_states.shape
    
        q = self.norm(self.W_Q(hidden_states))
        k = self.norm(self.W_K(hidden_states))
        v = self.W_V(hidden_states)

        q = q.reshape([b, m, self.n_h, self.d_h]).transpose(1, 2)
        k = k.reshape([b, m, self.n_h, self.d_h]).transpose(1, 2)
        v = v.reshape([b, m, self.n_h, self.d_h]).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, rope_rotations)

        attn_out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(b, m, d)
        out = self.W_O(attn_out)

        return out

# %% Sanity check
config = A2ModelConfig(
    vocab_size=1000,
    hidden_size=100,
    intermediate_size=200,
    num_attention_heads=4,
    rope_theta=1e4
)

x = torch.randn(2, 10, 100)
ids = torch.randn(2, 10, 100)
attention = A2Attention(config)
rotary_embedding = A2RotaryEmbedding(config)
rotary_embed = rotary_embedding(ids)
xout = attention(x, rotary_embed)
print(xout.shape, xout)

# %%


class A2DecoderLayer(nn.Module):
    """A complete Transformer decoder layer."""
    def __init__(self, config):
        super().__init__()
        self.attention = A2Attention(config)
        self.mlp = A2MLP(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, rope_rotations):
        att = self.attention(hidden_states, rope_rotations)
        # Normalization and residual connection from input
        att = self.norm(att) + hidden_states
        mlp = self.mlp(att)
        # Residual connection from attention output
        return  self.norm(mlp) + att


class A2Transformer(PreTrainedModel):
    """A language model based on the Transformer architecture."""
    
    config_class = A2ModelConfig

    def __init__(self, config):
        super().__init__(config)

        self.rotary_emb = A2RotaryEmbedding(config)
        # transformer decoder layers in a ModuleList for proper gradients.
        self.layers = nn.ModuleList([A2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.unembedding = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # This line should be called after you have set up all components.
        self.post_init()


    def forward(self, input_ids):
        rope_rotations = self.rotary_emb(input_ids) # pass this to all the transformer decoder layers

        # Call embedding, transformer decoder layers, last normalizer, and unembedding.
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, rope_rotations)
        hidden_states = self.norm(hidden_states)
        logits = self.unembedding(hidden_states)
        return logits

#### RoPE implementation (copied and simplified from HuggingFace). ####

def apply_rotary_pos_emb(q, k, rope_rotations, unsqueeze_dim=1):
    """Applies precomputed RoPE rotations to the query and key representations."""
    assert(q.shape == k.shape)
    assert(len(q.shape) == 4)
    cos, sin = rope_rotations
    assert(q.shape[2] == cos.shape[1])
    assert(q.shape[3] == cos.shape[2])    
    q_type, k_type = q.dtype, k.dtype
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q_type), k_embed.to(k_type)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class A2RotaryEmbedding(nn.Module):
    """RoPE position representation for use in Transformer attention."""

    def __init__(self, config, device=None):
        super().__init__()
        rope_theta = config.rope_theta
        head_dim = config.hidden_size // config.num_attention_heads
        partial_rotary_factor = 1.0
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))

    @torch.no_grad()
    def forward(self, x):
        position_ids = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos, sin

def generate(model: A2Transformer, prompt: str, tokenizer, max_length: int = 50, topk: int = 5, temperature: float = 1.0):
    model.eval()
    inputs = tokenizer([prompt], padding = True, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"][:, :-1]
    generated = input_ids
    print(generated)
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(generated)[:, -1, :]
            print(logits.shape)
            if topk is not None:
               # take the top-k tokens only
               logits, _ = torch.topk(logits, topk, dim=-1)
             
            distr = Categorical(logits=logits / temperature)
            next_token = distr.sample().unsqueeze(0)
            print(generated, next_token)
            generated = torch.cat((generated, next_token), dim=-1)
            if next_token == tokenizer.eos_token_id:
                break
    # convert ids to text 
    text = [tokenizer.inv_vocabulary[id] for id in generated[0].cpu().numpy()]
    return text
# %%
import sys
sys.path.append("..")
from a1.A1_skeleton import A1Tokenizer
import a1.A1_skeleton as a1mod

tokenizer = A1Tokenizer.from_file("../a1/tokenizer.pkl")

config = A2ModelConfig(
    vocab_size=len(tokenizer),
    hidden_size=100,
    intermediate_size=200,
    num_attention_heads=10,
    num_hidden_layers=1,
    rope_theta=1
)
model = A2Transformer(config)
X = torch.tensor([[1, 2, 3]], dtype=torch.long)
#print(X, X.shape)
sys.modules['__main__'].A1Tokenizer = a1mod.A1Tokenizer
sys.modules['__main__'].lowercase_tokenizer = a1mod.lowercase_tokenizer


out = model(X)
prompt = "He"
generated_out = generate(model, prompt, tokenizer)
#print(out.shape, out)
print(generated_out)
# %%
