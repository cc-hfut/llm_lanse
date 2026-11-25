import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import os
from common.formal_outputs import save_metrics, export_geo_head_top1
import numpy as np  # Use NumPy for metric calculation.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_results', 'RotatE')
os.makedirs(RESULT_DIR, exist_ok=True)
log = open(os.path.join(RESULT_DIR, 'RotatE.log'), 'w', encoding='utf-8')


# Load (h, r, t) triples and map names to IDs.
def load_triples(path, ent2id, rel2id):
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t = line.strip().split('\t')  # Split tab-separated head, relation, and tail fields.
            if h in ent2id and t in ent2id and r in rel2id:  # Check that all names have ID mappings.
                # Append the ID-based triple.
                triples.append((ent2id[h], rel2id[r], ent2id[t]))
    return triples


# Load entity or relation ID mappings, such as entity2id.txt.
def load_mapping(path):
    mapping = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            name, idx = line.strip().split('\t')  # Read the name and its ID.
            mapping[name] = int(idx)  # Convert the ID to an integer.
    return mapping


# Define the RotatE model.
class RotatE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100, margin=6.0, p_norm=1):
        super().__init__()
        self.dim = dim
        self.margin = margin
        self.p_norm = p_norm

        # Complex entity embeddings consist of real and imaginary parts.
        self.ent_emb_real = nn.Embedding(num_entities, dim)
        self.ent_emb_imag = nn.Embedding(num_entities, dim)

        # Relations use a rotation phase vector: the angle of a unit complex number.
        self.rel_phase = nn.Embedding(num_relations, dim)
        self.qwen_dim = 100
        self._init_weights()

    def _init_weights(self):
        r = 6.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.ent_emb_real.weight, -r, r)
        nn.init.uniform_(self.ent_emb_imag.weight, -r, r)
        nn.init.uniform_(self.rel_phase.weight, 0, 2 * math.pi)  # θ ∈ [0, 2π]

    def forward(self, h_idx, r_idx, t_idx):
        # Entity embeddings
        h_re = self.ent_emb_real(h_idx)
        h_im = self.ent_emb_imag(h_idx)
        t_re = self.ent_emb_real(t_idx)
        t_im = self.ent_emb_imag(t_idx)
        # Relation phases mapped to unit complex numbers
        phase = self.rel_phase(r_idx)
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)

        # Complex multiplication: h * r
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re

        # Compute differences.
        re_diff = hr_re - t_re
        im_diff = hr_im - t_im

        # Concatenate to [batch, 2*dim] and compute the norm.
        score = torch.norm(torch.cat([re_diff, im_diff], dim=1), p=self.p_norm, dim=1)
        return score

    def loss_fn(self, pos_score, neg_score):
        return torch.relu(pos_score + self.margin - neg_score).mean()

    def rotate_loss(pos_score, neg_score, adversarial_temperature=1.0):
        # Positive-sample loss
        positive_loss = -torch.log(torch.sigmoid(-pos_score)).mean()

        # Negative-sample loss with adversarial weights
        negative_scores = -neg_score
        negative_weights = torch.softmax(negative_scores * adversarial_temperature, dim=0).detach()
        negative_loss = -(negative_weights * torch.log(torch.sigmoid(negative_scores))).sum()

        return (positive_loss + negative_loss) / 2
    # ------------------ Forward pass with cached embeddings ------------------
    def forward_with_embeddings(self, h_re_emb, h_im_emb, r_phase_emb, t_re_emb, t_im_emb):
        'Run a forward pass with cached complex embeddings.\n\nArgs:\n    h_re_emb: Head real components [B, dim].\n    h_im_emb: Head imaginary components [B, dim].\n    r_phase_emb: Relation phases [B, dim].\n    t_re_emb: Tail real components [B, dim].\n    t_im_emb: Tail imaginary components [B, dim].\nReturns:\n    Distance scores [B]; lower is better.'
        # Relation phases mapped to unit complex numbers
        r_re = torch.cos(r_phase_emb)
        r_im = torch.sin(r_phase_emb)

        # Complex multiplication: h * r
        hr_re = h_re_emb * r_re - h_im_emb * r_im
        hr_im = h_re_emb * r_im + h_im_emb * r_re

        # Compute differences.
        re_diff = hr_re - t_re_emb
        im_diff = hr_im - t_im_emb

        # Concatenate to [batch, 2*dim] and compute the norm.
        score = torch.norm(torch.cat([re_diff, im_diff], dim=1), p=self.p_norm, dim=1)
        return score


# Generate negative triples by randomly replacing heads or tails.
def generate_negative_batch(batch, num_entities):
    neg_batch = []
    for h, r, t in batch:
        if random.random() < 0.5:
            h = random.randint(0, num_entities - 1)  # Replace the head.
        else:
            t = random.randint(0, num_entities - 1)  # Replace the tail.
        neg_batch.append((h, r, t))
    return neg_batch


# ------------------ Validation loss ------------------
def evaluate_loss_on_validation(model, val_data, num_entities, batch_size=128):
    'Compute the mean validation loss.'
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            pos_batch = val_data[i:i + batch_size]
            neg_batch = generate_negative_batch(pos_batch, num_entities)
            pos = torch.LongTensor(pos_batch).T.to(device)
            neg = torch.LongTensor(neg_batch).T.to(device)
            pos_score = model(*pos)
            neg_score = model(*neg)
            loss = model.loss_fn(pos_score, neg_score)
            total_loss += loss.item()
            num_batches += 1
    model.train()  # Restore training mode.
    return total_loss / num_batches if num_batches > 0 else float('inf')


# Training loop
def train(model, optimizer, train_data, val_data, num_entities, batch_size=128, epochs=1000,
          save_path="best_RotatE.pt", patience=30):
    model.to(device)
    best_val_loss = float("inf")  # Use validation loss for early stopping.
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(1, epochs + 1):
        if early_stop:
            print(f"Early stopping at epoch {epoch}")
            log.write(f"Early stopping at epoch {epoch}\n")
            break

        random.shuffle(train_data)  # Shuffle training triples each epoch.
        total_loss = 0
        for i in range(0, len(train_data), batch_size):  # Train in batches.
            pos_batch = train_data[i:i + batch_size]  # Positive batch
            neg_batch = generate_negative_batch(pos_batch, num_entities)  # Generate negative samples.

            pos = torch.LongTensor(pos_batch).T.to(device)  # Move tensors to the model device.
            neg = torch.LongTensor(neg_batch).T.to(device)  # Move tensors to the model device.

            pos_score = model(*pos)  # Score positive triples.
            neg_score = model(*neg)  # Score negative triples.
            loss = model.loss_fn(pos_score, neg_score)  # Compute the loss.

            optimizer.zero_grad()  # Clear gradients.
            loss.backward()  # Backpropagate gradients.
            optimizer.step()  # Update parameters.
            total_loss += loss.item()

        avg_train_loss = total_loss / (len(train_data) // batch_size)

        # ------------------ Validation ------------------
        val_loss = evaluate_loss_on_validation(model, val_data, num_entities, batch_size)
        print(f"Epoch {epoch}: Avg Train Loss = {avg_train_loss:.4f}, Avg Val Loss = {val_loss:.4f}")
        log.write(f"Epoch {epoch}: Avg Train Loss = {avg_train_loss:.4f}, Avg Val Loss = {val_loss:.4f}\n")

        # ------------------ Early stopping based on validation loss ------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the checkpoint with the lowest validation loss.
            torch.save(model.state_dict(), save_path)
            print(f"✔ Saved best model (based on val loss) at epoch {epoch} with val loss {val_loss:.4f}")
            log.write(f"✔ Saved best model (based on val loss) at epoch {epoch} with val loss {val_loss:.4f}\n")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Validation loss didn't improve for {patience} epochs. Early stopping...")
                log.write(f"Validation loss didn't improve for {patience} epochs. Early stopping...\n")
                early_stop = True

    # Restore the best checkpoint after training.
    try:
        model.load_state_dict(torch.load(save_path, map_location=device))
        print(f"Loaded best model from {save_path}")
        log.write(f"Loaded best model from {save_path}\n")
    except FileNotFoundError:
        print(f"Warning: Best model file {save_path} not found. Continuing with final model state.")
        log.write(f"Warning: Best model file {save_path} not found. Continuing with final model state.\n")


# Split triples into training, validation, and test sets.
def split_data(triples, train_ratio=0.9, val_ratio=0.09):
    random.shuffle(triples)
    n = len(triples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_data = triples[:train_end]
    val_data = triples[train_end:val_end]
    test_data = triples[val_end:]
    return train_data, val_data, test_data


# ------------------ Batched evaluation ------------------
def precompute_embeddings(model, ent_total, rel_total):
    'Precompute entity and relation embeddings.\n\nArgs:\n    model: A RotatE model.\n    ent_total: Number of entities.\n    rel_total: Number of relations.\nReturns:\n    Entity real and imaginary tensors and relation phases,\n    each shaped [total, dim].'
    model.eval()
    with torch.no_grad():
        # Create tensors for all entity and relation IDs.
        all_ent_ids = torch.arange(ent_total, device=device)  # [E]
        all_rel_ids = torch.arange(rel_total, device=device)  # [R]
        # Compute embeddings.
        ent_emb_real = model.ent_emb_real(all_ent_ids)  # [E, dim]
        ent_emb_imag = model.ent_emb_imag(all_ent_ids)  # [E, dim]
        rel_phase = model.rel_phase(all_rel_ids)  # [R, dim]
    return ent_emb_real, ent_emb_imag, rel_phase


def evaluate_hit_at_k_batch_optimized(model, test_triples, ent_total, rel_total, all_true_triples,
                                      batch_size=128, k_list=[1, 3, 10]):
    'Evaluate Hits@K in batches using cached entity and relation embeddings.\n\nArgs:\n    model: Trained model.\n    test_triples: Test triples [(h, r, t), ...].\n    ent_total: Number of entities.\n    rel_total: Number of relations.\n    all_true_triples: Known positive triples used for filtering.\n    batch_size: Evaluation batch size.\n    k_list: Values of K for Hits@K.'
    model.eval()
    hits = {k: 0 for k in k_list}
    ranks = []
    total = len(test_triples)
    # ------------------ Precompute entity and relation embeddings ------------------
    print("Precomputing entity and relation embeddings...")
    precomputed_ent_emb_real, precomputed_ent_emb_imag, precomputed_rel_phase = precompute_embeddings(model, ent_total,
                                                                                                      rel_total)  # [E, dim], [E, dim], [R, dim]
    print("Precomputation done.")

    with torch.no_grad():
        # Process test triples in batches.
        for i in range(0, total, batch_size):
            batch_triples = test_triples[i:i + batch_size]
            batch_size_current = len(batch_triples)
            # Build the batch ID tensors.
            h_batch_ids = torch.tensor([t[0] for t in batch_triples], device=device)  # [B]
            r_batch_ids = torch.tensor([t[1] for t in batch_triples], device=device)  # [B]
            t_batch_ids = torch.tensor([t[2] for t in batch_triples], device=device)  # [B]

            # ------------------ Index precomputed embeddings ------------------
            # Index precomputed embeddings by ID: [B, dim].
            h_batch_emb_real = precomputed_ent_emb_real[h_batch_ids]  # [B, dim]
            h_batch_emb_imag = precomputed_ent_emb_imag[h_batch_ids]  # [B, dim]
            r_batch_phase = precomputed_rel_phase[r_batch_ids]  # [B, dim]
            # Prepare embeddings for all tail candidates: [B, E, dim].
            t_candidate_embs_real = precomputed_ent_emb_real.unsqueeze(0).expand(batch_size_current, -1,
                                                                                 -1)  # [B, E, dim]
            t_candidate_embs_imag = precomputed_ent_emb_imag.unsqueeze(0).expand(batch_size_current, -1,
                                                                                 -1)  # [B, E, dim]

            # ------------------ Compute scores ------------------
            # Expand heads and relations: [B, dim] -> [B, E, dim].
            h_re_exp = h_batch_emb_real.unsqueeze(1).expand(-1, ent_total, -1)  # [B, E, dim]
            h_im_exp = h_batch_emb_imag.unsqueeze(1).expand(-1, ent_total, -1)  # [B, E, dim]
            r_phase_exp = r_batch_phase.unsqueeze(1).expand(-1, ent_total, -1)  # [B, E, dim]

            # Reshape to [B*E, dim] for batched computation.
            h_re_flat = h_re_exp.reshape(-1, h_re_exp.shape[-1])  # [B*E, dim]
            h_im_flat = h_im_exp.reshape(-1, h_im_exp.shape[-1])  # [B*E, dim]
            r_phase_flat = r_phase_exp.reshape(-1, r_phase_exp.shape[-1])  # [B*E, dim]
            t_re_flat = t_candidate_embs_real.reshape(-1, t_candidate_embs_real.shape[-1])  # [B*E, dim]
            t_im_flat = t_candidate_embs_imag.reshape(-1, t_candidate_embs_imag.shape[-1])  # [B*E, dim]

            # Score all (h, r, t_candidate) triples.
            scores_flat = model.forward_with_embeddings(h_re_flat, h_im_flat, r_phase_flat, t_re_flat,
                                                        t_im_flat)  # [B*E]
            scores = scores_flat.view(batch_size_current, ent_total)  # [B, E]

            # ------------------ Filter known triples ------------------
            filter_mask = torch.zeros_like(scores, dtype=torch.bool)  # [B, E]
            for idx, (h, r, t) in enumerate(batch_triples):
                # Filter known (h, r, t_other) positives while retaining the current target t.
                for t_candidate in range(ent_total):
                    if (h, r, t_candidate) in all_true_triples and t_candidate != t:
                        filter_mask[idx, t_candidate] = True
            # Set filtered distances to infinity so they cannot be selected.
            scores.masked_fill_(filter_mask, float('inf'))

            # ------------------ Sort candidates and compute ranks ------------------
            # Sort distances in ascending order; lower is better.
            _, sorted_indices = torch.sort(scores, dim=1, descending=False)  # [B, E]
            # Find the rank of each correct tail using vectorized operations.
            expanded_t_ids = t_batch_ids.unsqueeze(1)  # [B, 1]
            matches = (sorted_indices == expanded_t_ids)  # [B, E] bool
            # Find the first True column in each row.
            rank_indices = matches.float().argmax(dim=1)  # [B] (0-based index)
            ranks_batch = (rank_indices + 1).cpu().numpy()  # [B] (1-based rank)
            ranks.extend(ranks_batch.tolist())

            # Compute Hits@K using vectorized operations.
            for k in k_list:
                hits[k] += (ranks_batch <= k).sum().item()

    # Compute final metrics.
    hit_results = {k: hits[k] / total for k in k_list}
    mrr = np.mean(1.0 / np.array(ranks))
    mr = np.mean(ranks)
    # Report results.
    print(f"MR: {mr:.4f}")
    print(f"MRR: {mrr:.4f}")
    for k in k_list:
        print(f"Hit@{k}: {hit_results[k]:.4f}")
        log.write(f"Hit@{k}: {hit_results[k]:.4f}\n")
    log.write(f"MR: {mr:.4f}\n")
    log.write(f"MRR: {mrr:.4f}\n")
    return hit_results, mr, mrr


# Main entry point
if __name__ == '__main__':
    ent2id = load_mapping(r"../data/entity2id.txt")
    rel2id = load_mapping(r"../data/rel2id.txt")
    triples = load_triples(r"../data/landslide.txt", ent2id, rel2id)

    train_data = load_triples(r"../data/train.txt", ent2id, rel2id)
    val_data = load_triples(r"../data/valid.txt", ent2id, rel2id)
    test_data = load_triples(r"../data/test.txt", ent2id, rel2id)

    # Build the known-positive triple set for filtered evaluation.
    all_true_triples = set(train_data + val_data + test_data)

    model = RotatE(
        num_entities=len(ent2id) + 2,
        num_relations=len(rel2id) + 2,
        dim=100,
        p_norm=1,
        margin=1.0
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model.
    train(model, optimizer, train_data, val_data, num_entities=len(ent2id), epochs=1000,
          save_path=os.path.join(RESULT_DIR, "best_RotatE.pt"), patience=30)

    # Load the best checkpoint for final testing.
    # The train function already attempts to restore the best checkpoint.
    print("=== Testing on Test Set (Optimized Batch Evaluation with Precomputed Embeddings) ===")
    hits, mr, mrr = evaluate_hit_at_k_batch_optimized(
        model,
        test_data,
        ent_total=len(ent2id),
        rel_total=len(rel2id),  # Pass the relation count.
        all_true_triples=all_true_triples,
        batch_size=64,  # Adjust according to available GPU memory.
        k_list=[1, 3, 10]
    )
    save_metrics(RESULT_DIR, 'RotatE', hits, mr, mrr)
    export_geo_head_top1(model, 'RotatE', test_data, all_true_triples, ent2id, rel2id, RESULT_DIR, device)
    log.close()
