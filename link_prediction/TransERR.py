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
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_results', 'TransERR')
os.makedirs(RESULT_DIR, exist_ok=True)
log = open(os.path.join(RESULT_DIR, 'TransERR.log'), 'w', encoding='utf-8')


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


# Define the TransERR model.
class TransERR(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100, margin=6.0, p_norm=1):
        super().__init__()
        self.dim = dim
        self.margin = margin
        self.p_norm = p_norm

        # Entity embedding vectors
        self.ent_emb = nn.Embedding(num_entities, dim)

        # Relation direction represented by a unit vector.
        self.rel_dir = nn.Embedding(num_relations, dim)
        # Relation offset represented by a translation vector.
        self.rel_bias = nn.Embedding(num_relations, dim)
        self.qwen_dim = 100
        self._init_weights()

    def _init_weights(self):
        r = 6.0 / math.sqrt(self.dim)
        nn.init.uniform_(self.ent_emb.weight, -r, r)
        nn.init.uniform_(self.rel_dir.weight, -1.0, 1.0)
        nn.init.uniform_(self.rel_bias.weight, -r, r)

    def entity_encoder(self, entity):
        entity_embedding = self.ent_emb(entity)
        return entity_embedding

    def forward(self, h_idx, r_idx, t_idx):
        h = self.entity_encoder(h_idx)  # Head entity
        t = self.entity_encoder(t_idx)  # Tail entity
        r_dir = self.rel_dir(r_idx)  # Relation direction vector
        r_dir = nn.functional.normalize(r_dir, p=2, dim=-1)  # Normalize to unit vectors.
        r_bias = self.rel_bias(r_idx)  # Relation offset

        # TransERR representation: direction transformation followed by an offset.
        h_trans = h * r_dir + r_bias

        # Use distance scores; lower is better.
        score = torch.norm(h_trans - t, p=self.p_norm, dim=1)
        return score

    def loss_fn(self, pos_score, neg_score):
        return torch.relu(pos_score + self.margin - neg_score).mean()

    # ------------------ Forward pass with cached embeddings ------------------
    def forward_with_embeddings(self, h_emb, r_dir_emb_unit, r_bias_emb, t_emb):
        'Run a forward pass with precomputed embeddings.\n\nArgs:\n    h_emb: Head embeddings [B, dim].\n    r_dir_emb_unit: Normalized relation directions [B, dim].\n    r_bias_emb: Relation offsets [B, dim].\n    t_emb: Tail embeddings [B, dim].\nReturns:\n    Distance scores [B]; lower is better.'
        # TransERR representation: direction transformation followed by an offset.
        h_trans = h_emb * r_dir_emb_unit + r_bias_emb

        # Use distance scores; lower is better.
        score = torch.norm(h_trans - t_emb, p=self.p_norm, dim=1)
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
          save_path="best_TransERR.pt", patience=30):
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

        avg_train_loss = total_loss / (len(train_data) // batch_size)  # Compute the mean loss.

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


# ------------------ Optimized batched evaluation ------------------
def precompute_embeddings(model, ent_total, rel_total):
    'Precompute entity and relation embeddings.\n\nArgs:\n    model: A TransERR model.\n    ent_total: Number of entities.\n    rel_total: Number of relations.\nReturns:\n    Entity embeddings, normalized relation directions, and relation offsets,\n    each shaped [total, dim].'
    model.eval()
    with torch.no_grad():
        # Create tensors for all entity and relation IDs.
        all_ent_ids = torch.arange(ent_total, device=device)  # [E]
        all_rel_ids = torch.arange(rel_total, device=device)  # [R]
        # Compute embeddings.
        ent_emb = model.entity_encoder(all_ent_ids)  # [E, dim]
        rel_dir_emb = model.rel_dir(all_rel_ids)  # [R, dim]
        rel_bias_emb = model.rel_bias(all_rel_ids)  # [R, dim]
        # Normalize relation direction vectors.
        rel_dir_unit_emb = nn.functional.normalize(rel_dir_emb, p=2, dim=-1)  # [R, dim]
    return ent_emb, rel_dir_unit_emb, rel_bias_emb


def evaluate_hit_at_k_batch_optimized(model, test_triples, ent_total, rel_total, all_true_triples,
                                      batch_size=128, k_list=[1, 3, 10]):
    'Evaluate Hits@K in batches using cached entity and relation embeddings.\n\nArgs:\n    model: Trained model.\n    test_triples: Test triples [(h, r, t), ...].\n    ent_total: Number of entities.\n    rel_total: Number of relations.\n    all_true_triples: Known positive triples used for filtering.\n    batch_size: Evaluation batch size.\n    k_list: Values of K for Hits@K.'
    model.eval()
    hits = {k: 0 for k in k_list}
    ranks = []
    total = len(test_triples)
    # ------------------ Precompute entity and relation embeddings ------------------
    print("Precomputing entity and relation embeddings...")
    precomputed_ent_emb, precomputed_rel_dir_unit_emb, precomputed_rel_bias_emb = precompute_embeddings(model,
                                                                                                        ent_total,
                                                                                                        rel_total)  # [E, dim], [R, dim], [R, dim]
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
            h_batch_emb = precomputed_ent_emb[h_batch_ids]  # [B, dim]
            r_batch_dir_unit_emb = precomputed_rel_dir_unit_emb[r_batch_ids]  # [B, dim]
            r_batch_bias_emb = precomputed_rel_bias_emb[r_batch_ids]  # [B, dim]
            # Prepare embeddings for all tail candidates: [B, E, dim].
            t_candidate_embs = precomputed_ent_emb.unsqueeze(0).expand(batch_size_current, -1, -1)  # [B, E, dim]

            # ------------------ Compute scores ------------------
            # Expand heads, relation directions, and offsets: [B, dim] -> [B, E, dim].
            h_exp = h_batch_emb.unsqueeze(1).expand(-1, ent_total, -1)  # [B, E, dim]
            r_dir_unit_exp = r_batch_dir_unit_emb.unsqueeze(1).expand(-1, ent_total, -1)  # [B, E, dim]
            r_bias_exp = r_batch_bias_emb.unsqueeze(1).expand(-1, ent_total, -1)  # [B, E, dim]

            # Reshape to [B*E, dim] for batched computation.
            h_flat = h_exp.reshape(-1, h_exp.shape[-1])  # [B*E, dim]
            r_dir_unit_flat = r_dir_unit_exp.reshape(-1, r_dir_unit_exp.shape[-1])  # [B*E, dim]
            r_bias_flat = r_bias_exp.reshape(-1, r_bias_exp.shape[-1])  # [B*E, dim]
            t_flat = t_candidate_embs.reshape(-1, t_candidate_embs.shape[-1])  # [B*E, dim]

            # Score all (h, r, t_candidate) triples.
            scores_flat = model.forward_with_embeddings(h_flat, r_dir_unit_flat, r_bias_flat, t_flat)  # [B*E]
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

    model = TransERR(
        num_entities=len(ent2id) + 2,
        num_relations=len(rel2id) + 2,
        dim=100,
        p_norm=1,
        margin=1.0
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model.
    train(model, optimizer, train_data, val_data, num_entities=len(ent2id), epochs=1000,
          save_path=os.path.join(RESULT_DIR, "best_TransERR.pt"), patience=30)

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
    save_metrics(RESULT_DIR, 'TransERR', hits, mr, mrr)
    export_geo_head_top1(model, 'TransERR', test_data, all_true_triples, ent2id, rel2id, RESULT_DIR, device)
    log.close()
