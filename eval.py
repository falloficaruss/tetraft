import torch


def model_device(model) -> torch.device:
    """Best-effort device for a (possibly device_map-sharded) model."""
    if hasattr(model, "device") and model.device is not None:
        try:
            return torch.device(model.device)
        except (TypeError, RuntimeError):
            pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def evaluate_perplexity(model, dataloader, max_batches=None, device=None):
    """Token-weighted mean NLL → perplexity on a causal-LM dataloader.

    Expects batches with ``input_ids`` and ``labels`` (and optional
    ``attention_mask``). Labels use ``-100`` for ignored positions.
    """
    model.eval()
    if device is None:
        device = model_device(model)

    total_loss = 0.0
    total_tokens = 0
    n_batches = 0

    for batch in dataloader:
        if max_batches is not None and n_batches >= max_batches:
            break

        batch = {
            k: v.to(device) if hasattr(v, "to") else v
            for k, v in batch.items()
        }
        outputs = model(**batch)
        loss = outputs.loss

        labels = batch.get("labels", batch["input_ids"])
        batch_tokens = (labels != -100).sum().item()
        # HF causal LM loss is already mean over non-ignored tokens.
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
        n_batches += 1

    model.train()
    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss, dtype=torch.float64)).item()
    return perplexity
