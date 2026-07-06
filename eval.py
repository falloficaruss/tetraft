import torch


@torch.no_grad()
def evaluate_perplexity(model, dataloader, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in dataloader:
        if max_batches is not None and n_batches >= max_batches:
            break

        batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        batch_tokens = (batch.get("labels", batch["input_ids"]) != -100).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        n_batches += 1

    model.train()
    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity
