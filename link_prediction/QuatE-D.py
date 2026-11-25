import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import json
import os
from common.formal_outputs import save_metrics, export_geo_head_top1
import numpy as np  # Use NumPy for metric calculation.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Ensure that the log path is valid.
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_results', 'QuatE-D')
os.makedirs(RESULT_DIR, exist_ok=True)
log = open(os.path.join(RESULT_DIR, 'QuatE-D.log'), 'w', encoding='utf-8')


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


# Define the QuatED model.
class QuatED(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100, margin=6.0, p_norm=1):
        super().__init__()
        self.dim = dim
        self.margin = margin
        self.p_norm = p_norm

        # Each entity quaternion has real, i, j, and k components.
        self.ent_r = nn.Embedding(num_entities, dim)
        self.ent_i = nn.Embedding(num_entities, dim)
        self.ent_j = nn.Embedding(num_entities, dim)
        self.ent_k = nn.Embedding(num_entities, dim)

        # Relation quaternions and learnable magnitude parameters
        self.rel_r = nn.Embedding(num_relations, dim)
        self.rel_i = nn.Embedding(num_relations, dim)
        self.rel_j = nn.Embedding(num_relations, dim)
        self.rel_k = nn.Embedding(num_relations, dim)
        self.rel_mod = nn.Embedding(num_relations, 1)  # Magnitude component
        self.qwen_dim = 100
        self._init_weights()

    def _init_weights(self):
        r = 6.0 / math.sqrt(self.dim)
        for emb in [self.ent_r, self.ent_i, self.ent_j, self.ent_k,
                    self.rel_r, self.rel_i, self.rel_j, self.rel_k]:
            nn.init.uniform_(emb.weight, -r, r)
        nn.init.ones_(self.rel_mod.weight)  # Initialize magnitudes to one.


    # Quaternion multiplication returns one real and three imaginary components.
    def quaternion_mul(self, h, r):
        hr, hi, hj, hk = h
        rr, ri, rj, rk = r
        qr = rr * hr - ri * hi - rj * hj - rk * hk
        qi = rr * hi + ri * hr + rj * hk - rk * hj
        qj = rr * hj - ri * hk + rj * hr + rk * hi
        qk = rr * hk + ri * hj - rj * hi + rk * hr
        return qr, qi, qj, qk

    def forward(self, h_idx, r_idx, t_idx):
       
        # Entity quaternions
        h = (self.ent_r(h_idx) , self.ent_i(h_idx) ,
             self.ent_j(h_idx) , self.ent_k(h_idx) )
        t = (self.ent_r(t_idx) , self.ent_i(t_idx) ,
             self.ent_j(t_idx) , self.ent_k(t_idx) )

        # Look up quaternion relation embeddings.
        r = (self.rel_r(r_idx), self.rel_i(r_idx), self.rel_j(r_idx), self.rel_k(r_idx))

        # Normalize relations to retain rotation direction.
        r_norm = torch.sqrt(sum([x ** 2 for x in r]) + 1e-9)
        r_unit = tuple(x / r_norm for x in r)

        # Quaternion multiplication: head entity times unit relation.
        h_r = self.quaternion_mul(h, r_unit)

        # Apply magnitude scaling.
        rel_mod = self.rel_mod(r_idx)  # shape: [B, 1]
        h_r_scaled = [v * rel_mod for v in h_r]

        # Subtract the tail entity.
        diff = [(h_r_scaled[i] - t[i]) for i in range(4)]

        # Concatenate components and compute the L1/L2 norm.
        score = torch.norm(torch.cat(diff, dim=1), p=self.p_norm, dim=1)
        return score

    def loss_fn(self, pos_score, neg_score):
        return torch.relu(pos_score + self.margin - neg_score).mean()

    # ------------------ Forward pass with cached embeddings ------------------
    def forward_with_embeddings(self, h_emb, r_emb_unit, r_mod_emb, t_emb):
        'Run a forward pass with precomputed quaternion embeddings.\n\nArgs:\n    h_emb: Four head components, each shaped [B, dim].\n    r_emb_unit: Four normalized relation components, each shaped [B, dim].\n    r_mod_emb: Relation magnitudes shaped [B, 1].\n    t_emb: Four tail components, each shaped [B, dim].\nReturns:\n    Distance scores shaped [B]; lower is better.'
        # Quaternion multiplication: head entity times unit relation.
        h_r = self.quaternion_mul(h_emb, r_emb_unit)

        # Apply magnitude scaling.
        h_r_scaled = [v * r_mod_emb for v in h_r]

        # Subtract the tail entity.
        diff = [(h_r_scaled[i] - t_emb[i]) for i in range(4)]

        # Concatenate components and compute the L1/L2 norm.
        score = torch.norm(torch.cat(diff, dim=1), p=self.p_norm, dim=1)
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
          save_path="best_QuatED.pt", patience=30):
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
def precompute_entity_embeddings(model, ent_total):
    'Precompute entity quaternion embeddings.\n\nArgs:\n    model: A QuatED model.\n    ent_total: Number of entities.\nReturns:\n    Four component tensors, each shaped [ent_total, dim].'
    model.eval()
    with torch.no_grad():
        # Create a tensor of all entity IDs.
        all_ent_ids = torch.arange(ent_total, device=device)  # [E]
    
        # Compute and combine the base quaternion embeddings.
        ent_r_emb = model.ent_r(all_ent_ids)   # [E, dim]
        ent_i_emb = model.ent_i(all_ent_ids)   # [E, dim]
        ent_j_emb = model.ent_j(all_ent_ids)   # [E, dim]
        ent_k_emb = model.ent_k(all_ent_ids)   # [E, dim]
    return (ent_r_emb, ent_i_emb, ent_j_emb, ent_k_emb)


def precompute_relation_embeddings(model, rel_total):
    'Precompute relation quaternion embeddings and magnitudes.\n\nArgs:\n    model: A QuatED model.\n    rel_total: Number of relations.\nReturns:\n    Four normalized component tensors shaped [rel_total, dim]\n    and a magnitude tensor shaped [rel_total, 1].'
    model.eval()
    with torch.no_grad():
        all_rel_ids = torch.arange(rel_total, device=device)  # [R]
        # Look up the unnormalized relation embeddings.
        rel_r_emb = model.rel_r(all_rel_ids)  # [R, dim]
        rel_i_emb = model.rel_i(all_rel_ids)  # [R, dim]
        rel_j_emb = model.rel_j(all_rel_ids)  # [R, dim]
        rel_k_emb = model.rel_k(all_rel_ids)  # [R, dim]
        # Compute magnitudes.
        rel_mod_emb = model.rel_mod(all_rel_ids)  # [R, 1]
        # Normalize quaternion relation embeddings.
        r_norm = torch.sqrt(rel_r_emb ** 2 + rel_i_emb ** 2 + rel_j_emb ** 2 + rel_k_emb ** 2 + 1e-9)  # [R, dim]
        rel_r_emb_unit = rel_r_emb / r_norm  # [R, dim]
        rel_i_emb_unit = rel_i_emb / r_norm  # [R, dim]
        rel_j_emb_unit = rel_j_emb / r_norm  # [R, dim]
        rel_k_emb_unit = rel_k_emb / r_norm  # [R, dim]
    return (rel_r_emb_unit, rel_i_emb_unit, rel_j_emb_unit, rel_k_emb_unit, rel_mod_emb)


def evaluate_hit_at_k_batch_optimized(model, test_triples, ent_total, rel_total, all_true_triples,
                                      batch_size=128, k_list=[1, 3, 10]):
    'Evaluate Hits@K in batches using cached entity and relation embeddings.\n\nArgs:\n    model: Trained model.\n    test_triples: Test triples [(h, r, t), ...].\n    ent_total: Number of entities.\n    rel_total: Number of relations.\n    all_true_triples: Known positive triples used for filtering.\n    batch_size: Evaluation batch size.\n    k_list: Values of K for Hits@K.'
    model.eval()
    hits = {k: 0 for k in k_list}
    ranks = []
    total = len(test_triples)
    # ------------------ Precompute entity and relation embeddings ------------------
    print("Precomputing entity embeddings...")
    precomputed_ent_embs = precompute_entity_embeddings(model, ent_total)  # tuple of 4 tensors [E, dim]
    print("Precomputing relation embeddings...")
    precomputed_rel_embs = precompute_relation_embeddings(model, rel_total)  # tuple of 5 tensors ([R, dim] x4, [R, 1])
    # Unpack relation embeddings.
    precomputed_rel_quat_embs_unit = precomputed_rel_embs[:4]  # (rr_unit, ri_unit, rj_unit, rk_unit) each [R, dim]
    precomputed_rel_mod_embs = precomputed_rel_embs[4]  # [R, 1]
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
            # Index cached embeddings by ID: [B, dim] or [B, 1].
            h_batch_emb = tuple(emb[h_batch_ids] for emb in precomputed_ent_embs)  # tuple of 4 [B, dim]
            r_batch_emb_unit = tuple(emb[r_batch_ids] for emb in precomputed_rel_quat_embs_unit)  # tuple of 4 [B, dim]
            r_batch_mod_emb = precomputed_rel_mod_embs[r_batch_ids]  # [B, 1]
            # Prepare embeddings for all tail candidates: [B, E, dim].
            t_candidate_embs = tuple(emb.unsqueeze(0).expand(batch_size_current, -1, -1) for emb in
                                     precomputed_ent_embs)  # tuple of 4 [B, E, dim]

            # ------------------ Score cached embeddings with forward_with_embeddings ------------------
            # Expand heads and unit relations: [B, dim] -> [B, E, dim].
            h_exp = tuple(eb.unsqueeze(1).expand(-1, ent_total, -1) for eb in h_batch_emb)  # tuple of 4 [B, E, dim]
            r_unit_exp = tuple(
                eb.unsqueeze(1).expand(-1, ent_total, -1) for eb in r_batch_emb_unit)  # tuple of 4 [B, E, dim]
            # Expand relation magnitudes: [B, 1] -> [B, E, 1].
            r_mod_exp = r_batch_mod_emb.unsqueeze(1).expand(-1, ent_total, -1)  # [B, E, 1]

            # Reshape to [B*E, dim] or [B*E, 1] for batch scoring.
            h_flat = tuple(eb.reshape(-1, eb.shape[-1]) for eb in h_exp)  # tuple of 4 [B*E, dim]
            r_unit_flat = tuple(eb.reshape(-1, eb.shape[-1]) for eb in r_unit_exp)  # tuple of 4 [B*E, dim]
            r_mod_flat = r_mod_exp.reshape(-1, r_mod_exp.shape[-1])  # [B*E, 1]
            t_flat = tuple(eb.reshape(-1, eb.shape[-1]) for eb in t_candidate_embs)  # tuple of 4 [B*E, dim]

            # Score all (h, r, t_candidate) triples.
            scores_flat = model.forward_with_embeddings(h_flat, r_unit_flat, r_mod_flat, t_flat)  # [B*E]
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
    # Ensure that the path is valid.
    ent2id = load_mapping(r"../data/entity2id.txt")
    rel2id = load_mapping(r"../data/rel2id.txt")
    triples = load_triples(r"../data/landslide.txt", ent2id, rel2id)

    train_data = load_triples(r"../data/train.txt", ent2id, rel2id)
    val_data = load_triples(r"../data/valid.txt", ent2id, rel2id)
    test_data = load_triples(r"../data/test.txt", ent2id, rel2id)

    # Build the known-positive triple set for filtered evaluation.
    all_true_triples = set(train_data + val_data + test_data)

    model = QuatED(
        num_entities=len(ent2id) + 2,
        num_relations=len(rel2id) + 2,
        dim=100,
        p_norm=1,
        margin=1.0
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Pass validation data and early-stopping patience to train.
    train(model, optimizer, train_data, val_data, num_entities=len(ent2id), epochs=1000,
          save_path=os.path.join(RESULT_DIR, "best_QuatED.pt"), patience=30)

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
    save_metrics(RESULT_DIR, 'QuatE-D', hits, mr, mrr)
    export_geo_head_top1(model, 'QuatE-D', test_data, all_true_triples, ent2id, rel2id, RESULT_DIR, device)
    log.close()  # Close log handlers.
