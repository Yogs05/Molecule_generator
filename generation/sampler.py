import torch

def generate_smiles(
    model,
    vocab,
    max_length=100,
    temperature=1.0,
    device="cpu"
):
    model.eval()

    start_idx = vocab.start_idx
    end_idx = vocab.end_idx

    input_token = torch.tensor([[start_idx]], device=device)
    hidden = None

    generated = []

    with torch.no_grad():
        for _ in range(max_length):
            logits, hidden = model(input_token, hidden)

            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, 1).item()

            if next_idx == end_idx:
                break

            generated.append(next_idx)
            input_token = torch.tensor([[next_idx]], device=device)

    return vocab.decode(generated)
