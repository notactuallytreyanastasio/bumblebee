# Qwen3 Support in Bumblebee

This fork adds support for the Qwen3 model family to Bumblebee.

## Why This Fork Exists

Qwen3 is Alibaba's state-of-the-art LLM series, but upstream Bumblebee doesn't support it yet. This fork adds:

- `Bumblebee.Text.Qwen3` - Model architecture implementation
- Tokenizer support for Qwen3's vocabulary
- ChatML prompt format handling

## The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    Bumblebee Model Zoo                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Upstream Models:           This Fork Adds:                  │
│  ┌──────────────┐          ┌──────────────┐                 │
│  │    BERT      │          │    Qwen3     │ <-- NEW         │
│  │    GPT-2     │          └──────────────┘                 │
│  │    LLaMA     │                                           │
│  │    Gemma     │          Features:                        │
│  │    Mistral   │          - 0.5B to 235B parameter sizes   │
│  │    ...       │          - ChatML format                  │
│  └──────────────┘          - RoPE with base scaling         │
│                            - Grouped Query Attention         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Model Architecture

Qwen3 is a decoder-only transformer with:

| Component | Qwen3-8B Config |
|-----------|-----------------|
| Layers | 36 |
| Hidden size | 4096 |
| Attention heads | 32 |
| KV heads | 8 (GQA) |
| Vocab size | 151936 |
| Context length | 32768 |
| RoPE base | 1000000 |

### Key Differences from LLaMA

1. **Vocabulary**: 151936 tokens vs LLaMA's 32000
2. **Special tokens**: ChatML format (`<|im_start|>`, `<|im_end|>`)
3. **RoPE scaling**: Different base frequency
4. **MLP**: Uses SiLU activation (same as LLaMA)

## Usage

### Basic Text Generation

```elixir
{:ok, model} = Bumblebee.load_model({:hf, "Qwen/Qwen3-8B"})
{:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, "Qwen/Qwen3-8B"})

serving = Bumblebee.Text.Generation.build(model, tokenizer,
  max_new_tokens: 100,
  temperature: 0.7
)

Nx.Serving.run(serving, "Hello, how are you?")
```

### ChatML Format

Qwen3 uses ChatML for chat interactions:

```elixir
# Format a conversation
prompt = """
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
What is the capital of France?
<|im_end|>
<|im_start|>assistant
"""

# The model will generate the assistant's response
```

### With Quantized Models

For 4-bit models (requires EMLX quantization fork):

```elixir
# Load quantized weights
{:ok, model} = Bumblebee.load_model({:local, "/path/to/qwen3-8b-4bit"},
  backend: {EMLX.Backend, device: :gpu}
)
```

## Files Added

```
lib/bumblebee/text/
├── qwen3.ex                    # Main model module
└── text_reranking_qwen3.ex     # Reranking variant

test/bumblebee/text/
└── qwen3_test.exs              # Tests

notebooks/
└── qwen3.livemd                # Interactive notebook
```

## Model Variants

| Model | Parameters | Recommended Use |
|-------|------------|-----------------|
| Qwen3-0.5B | 0.5B | Edge devices, fast inference |
| Qwen3-1.5B | 1.5B | Balanced speed/quality |
| Qwen3-4B | 4B | Good quality, reasonable memory |
| Qwen3-8B | 8B | High quality, 16GB+ RAM |
| Qwen3-14B | 14B | Very high quality |
| Qwen3-32B | 32B | Near-frontier quality |
| Qwen3-72B | 72B | Frontier quality |

## Tokenizer Details

Qwen3 uses a SentencePiece-based tokenizer with:

- Byte-level BPE
- 151936 vocabulary size
- Special tokens for ChatML:
  - `<|im_start|>` (151644)
  - `<|im_end|>` (151645)
  - `<|endoftext|>` (151643)

## Implementation Notes

### RoPE (Rotary Position Embeddings)

```elixir
# Qwen3 uses a higher base frequency than LLaMA
rope_base = 1_000_000  # vs 10_000 for LLaMA

# This allows longer context windows
```

### Grouped Query Attention

```elixir
# 8B model: 32 query heads, 8 KV heads (4:1 ratio)
# This reduces memory for KV cache
num_heads = 32
num_kv_heads = 8
head_dim = hidden_size / num_heads  # 128
```

### Layer Normalization

Qwen3 uses RMSNorm (like LLaMA) rather than LayerNorm:

```elixir
def rms_norm(x, weight, eps \\ 1.0e-6) do
  variance = Nx.mean(Nx.pow(x, 2), axes: [-1], keep_axes: true)
  x = x * Nx.rsqrt(variance + eps)
  x * weight
end
```

## Branch

This is on the `feat/qwen3` branch:

```bash
git checkout feat/qwen3
```

## Testing

```bash
mix test test/bumblebee/text/qwen3_test.exs
```

## Upstream Contribution

This should be contributed back to upstream Bumblebee once stabilized. The implementation follows Bumblebee's patterns for consistency with other model families.

## Related

- [Qwen3 Model Card](https://huggingface.co/Qwen)
- [ChatML Format](https://github.com/openai/openai-python/blob/main/chatml.md)
- [EMLX Quantization](../emlx/QUANTIZATION.md)
