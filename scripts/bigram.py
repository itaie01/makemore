import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------
# set seeds
torch.manual_seed(42)
torch.mps.manual_seed(42)

# ------------------
# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "mps" if torch.mps.is_available() else "cpu"
device = "mps"
eval_iters = 200
n_embed = 32
# ------------------

with open("../data/shakespeare.txt", "r") as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
# print("".join(chars))

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}


def encode(s: str):
    return [stoi[c] for c in s]


def decode(idxs: list[int]):
    return [itos[i] for i in idxs]


# ------------------

data = torch.tensor(encode(text))

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

x = train_data[:block_size]
y = train_data[1 : block_size + 1]
# for t in range(block_size):
#     context = x[: t + 1]
#     target = y[t]
#     print(f"Target for input: {context} is {target}")


def get_batch(split):
    # generate small batch of data of inputs x and target y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


xb, yb = get_batch("train")
# for x, y in zip(xb, yb):
#     print(f"Targets for {x} are {y}")


# ------------------
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.do = nn.Dropout()

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)

        # compute attention
        wei = k @ q.transpose(-2, -1) * self.head_size**-0.5
        wei = torch.masked_fill(wei, self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.do(wei)

        # return weighted aggregation
        return wei @ v


# ------------------
class MultiAttentionHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttentionHead(head_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))


# ------------------
class Block(nn.Module):
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiAttentionHead(n_heads, head_size)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(
                4 * n_embed, n_embed
            ),  # attention going back into residual pathway
        )
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        return x + self.ffwd(self.ln2(x + self.sa(self.ln1(x))))


# ------------------
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        # each token reacs logits for text token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            Block(n_embed, n_heads=4),
            Block(n_embed, n_heads=4),
            Block(n_embed, n_heads=4),
        )
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(
                B * T
            )  # -1 if we want pytorch to guess what value should be
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in current context
        for _ in range(max_new_tokens):
            # cap idx at block_size
            idx_cap = idx[:, -block_size:]

            # get predictions
            logits, _ = self(idx_cap)

            # focus on last time step
            logits = logits[:, -1, :]

            # get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # sample from created distribution
            next_idx = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to running sequence
            idx = torch.cat((idx, next_idx), dim=1)  # (B, T + 1)

        return idx


# ------------------
# Training and Evaluating
torch.manual_seed(1337)
model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


@torch.inference_mode()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


model = model.to(device)
for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"Step {step}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f}"
        )

    # sample batch
    xb, yb = get_batch("train")

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
# ------------------


@torch.inference_mode()
def generate_text(legnth: int):
    model.to("cpu")
    model.eval()
    print(
        "".join(
            decode(
                model.generate(
                    idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=legnth
                )[0].tolist()
            )
        )
    )


generate_text(500)
