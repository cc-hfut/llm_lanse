# -*- coding: utf-8 -*-
import logging
import os
import random
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from llama_cpp import Llama

# The ablation copy reuses link_prediction output helpers; its model implementation remains local.
LINK_PREDICTION_DIR = Path(__file__).resolve().parents[1] / 'link_prediction'
if str(LINK_PREDICTION_DIR) not in sys.path:
    sys.path.insert(0, str(LINK_PREDICTION_DIR))
from common.formal_outputs import save_metrics

# ------------------ Configuration ------------------
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_results', 'LLM-GeoSE')

# Create the output directory.
os.makedirs(out_dir, exist_ok=True)
# Log file
log_file = os.path.join(out_dir, f"LLM_LanSE_{os.path.basename(out_dir)}.log")


def setup_logging(log_file, console=True, filemode='w'):
    logger = logging.getLogger()
    logger.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler() if console else logging.NullHandler()
        ]
    )
    return logger


setup_logging(log_file=log_file)

MODEL_NAME = "Qwen3-Embedding-8B-f16.gguf"
MODEL_PATH = os.environ.get(
    "QWEN3_EMBEDDING_MODEL",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Qwen3-Embedding-8B-GGUF", MODEL_NAME))
)

logging.info(f"Qwen3 model path: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found. Check the path: {MODEL_PATH}")
logging.info(f"File size: {os.path.getsize(MODEL_PATH) / (1024 * 1024):.2f} MB")

try:
    logging.info('Loading the model...')
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=512,
        n_threads=4,
        n_gpu_layers=20,
        embedding=True,
        verbose=False
    )
    logging.info('Model loaded successfully.')
except Exception as e:
    logging.info(f"Model loading failed: {str(e)}")
    raise
# Select the available device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")


# ------------------ Utilities ------------------
def load_mapping(path):
    'Load entity2id.txt or rel2id.txt.'
    name2id, id2name = {}, {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            name, idx = parts
            name2id[name] = int(idx)
            id2name[int(idx)] = name
    return name2id, id2name


def load_triples(path, ent2id, rel2id):
    'Load triples.'
    triples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            h, r, t = line.strip().split('\t')
            if h in ent2id and r in rel2id and t in ent2id:
                triples.append((ent2id[h], rel2id[r], ent2id[t]))
    return triples


# ------------------ Embedding cache ------------------
text_cache = {}
word_vectors_all = None
word_vectors_keywords = None
entid2keywords = None


def get_embedding_from_llm(text):
    'Generate embeddings using Qwen3-Embedding.'
    if llm is None:
        return None


def load_word2vec_resources(use_keywords=False):
    'Load ablation resources on demand; the full model does not require gensim.'
    global word_vectors_all, word_vectors_keywords, entid2keywords
    from gensim.models import KeyedVectors
    base = str(LINK_PREDICTION_DIR)
    data_dir = os.path.abspath(os.path.join(base, '..', 'data'))
    if use_keywords:
        path = os.path.join(base, 'word2vec_keywords.kv')
        mapping_path = os.path.join(data_dir, 'entityid2keywords.json')
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Missing keyword-ablation input: {mapping_path}")
        word_vectors_keywords = KeyedVectors.load(path, mmap='r')
        with open(mapping_path, 'r', encoding='utf-8') as f:
            entid2keywords = json.load(f)
    else:
        path = os.path.join(base, 'word2vec_all_words.kv')
        word_vectors_all = KeyedVectors.load(path, mmap='r')


def get_word2vec_embedding(text, keywords=False):
    vectors = word_vectors_keywords if keywords else word_vectors_all
    tokens = text if isinstance(text, list) else [str(text)]
    found = [torch.tensor(vectors[token], dtype=torch.float32, device=device)
             for token in tokens if token in vectors]
    if not found:
        return torch.zeros(vectors.vector_size, dtype=torch.float32, device=device)
    return torch.stack(found).mean(0)
    try:
        if not isinstance(text, str):
            text = str(text)
        if text in text_cache:
            return text_cache[text]

        response = llm.create_embedding(text)
        embedding = np.array(response['data'][0]['embedding'])
        text_cache[text] = torch.tensor(embedding, dtype=torch.float32).to(device)
        return torch.tensor(embedding, dtype=torch.float32).to(device)
    except Exception as e:
        logging.info(f"Embedding generation failed: {e}")
        return None


# ------------------ Model definition ------------------
class LLM_LanSE(nn.Module):
    def __init__(self, num_entities, num_relations, num_relation_types, dim=100, margin=1.0, p_norm=1,
                 ablation='full'):
        super().__init__()
        self.margin = margin
        self.p_norm = p_norm
        self.dim = dim
        self.encoder_dim = 1024
        self.ablation = ablation
        self.w2v_linear = nn.Linear(100, self.encoder_dim)
        # Concatenate name, description, temporal, and spatial embeddings and project them to quaternions.
        self.ent_r_linear = nn.Linear(self.encoder_dim * 4, dim)
        self.ent_i_linear = nn.Linear(self.encoder_dim * 4, dim)
        self.ent_j_linear = nn.Linear(self.encoder_dim * 4, dim)
        self.ent_k_linear = nn.Linear(self.encoder_dim * 4, dim)

        # Learn a quaternion representation for each relation type.
        self.relation_type_embeddings = nn.Embedding(num_relation_types, self.encoder_dim)
        # Relation projection
        self.rel_r_linear = nn.Linear(self.encoder_dim, dim)
        self.rel_i_linear = nn.Linear(self.encoder_dim, dim)
        self.rel_j_linear = nn.Linear(self.encoder_dim, dim)
        self.rel_k_linear = nn.Linear(self.encoder_dim, dim)
        # Generate all relation dual-quaternion components with a shared projection, then split into (r, i, j, k).
        self.rel_dual_linear = nn.Linear(self.encoder_dim, dim * 4)
        # Relation magnitude parameters
        self.rel_mod = nn.Embedding(num_relations, 1)

        self._init_weights()

    def _init_weights(self):
        linear_layers = [self.w2v_linear,
                         self.ent_r_linear, self.ent_i_linear, self.ent_j_linear, self.ent_k_linear,
                         self.rel_r_linear, self.rel_i_linear, self.rel_j_linear, self.rel_k_linear,
                         self.rel_dual_linear]
        for linear in linear_layers:
            if linear is not None:
                nn.init.xavier_uniform_(linear.weight)
                nn.init.zeros_(linear.bias)
        nn.init.xavier_uniform_(self.relation_type_embeddings.weight)

    def quat_multiply(self, h, r):
        'Quaternion multiplication.'
        hr, hi, hj, hk = h
        rr, ri, rj, rk = r
        qr = rr * hr - ri * hi - rj * hj - rk * hk
        qi = rr * hi + ri * hr + rj * hk - rk * hj
        qj = rr * hj - ri * hk + rj * hr + rk * hi
        qk = rr * hk + ri * hj - rj * hi + rk * hr
        return qr, qi, qj, qk

    def quat_conjugate(self, q):
        'Quaternion conjugation.'
        qr, qi, qj, qk = q
        return qr, -qi, -qj, -qk

    def quat_inner_product(self, q1, q2):
        'Quaternion inner product.'
        r1, i1, j1, k1 = q1
        r2, i2, j2, k2 = q2
        return r1 * r2 + i1 * i2 + j1 * j2 + k1 * k2

    def encode_entity(self, entity_ids, id2ent_name, id2ent_duc, id2time, id2location):
        'Fuse name, description, temporal, and spatial features into entity quaternions.'
        emb = []
        for id in entity_ids:
            idx = id.item()
            zero = torch.zeros(self.encoder_dim, dtype=torch.float32, device=device)
            if self.ablation == 'wo_entity_enhancement':
                name_embedding = self.w2v_linear(get_word2vec_embedding(id2ent_name[idx]))
                duc_embedding = time_embedding = location_embedding = zero
            else:
                name_embedding = get_embedding_from_llm(id2ent_name[idx])
                if self.ablation == 'wo_text_keywords':
                    keywords = entid2keywords.get(str(idx), [])
                    if isinstance(keywords, str):
                        keywords = keywords.split()
                    duc_embedding = self.w2v_linear(get_word2vec_embedding(keywords, keywords=True))
                elif self.ablation == 'wo_describe_text':
                    duc_embedding = zero
                else:
                    duc_embedding = get_embedding_from_llm(id2ent_duc[str(idx)])
                time_embedding = zero if self.ablation in ('wo_time_property', 'wo_time_space') \
                    else get_embedding_from_llm(id2time[str(idx)])
                location_embedding = zero if self.ablation in ('wo_spatial_property', 'wo_time_space') \
                    else get_embedding_from_llm(id2location[str(idx)])
            emb.append(torch.cat(
                (name_embedding, duc_embedding, time_embedding, location_embedding), dim=-1
            ))
        entity_embeding = torch.stack(emb).to(device)
        # Linear projection
        r = self.ent_r_linear(entity_embeding)
        i = self.ent_i_linear(entity_embeding)
        j = self.ent_j_linear(entity_embeding)
        k = self.ent_k_linear(entity_embeding)
        return r, i, j, k

    def encode_relation(self, relation_ids, id2rel_name):
        'Encode relations.'
        emb = []
        for id in relation_ids:
            idx = id.item()
            rel_embedding = get_embedding_from_llm(id2rel_name[idx])
            # Look up the relation type ID and its embedding.
            relation_type_id = relid2typeid[str(idx)]  # Look up the relation type ID.
            if self.ablation == 'wo_relation_enhancement':
                emb.append(rel_embedding)
            else:
                relation_type_vectors = self.relation_type_embeddings(
                    torch.tensor([relation_type_id], device=device))  # [1, encoder_dim]
                emb.append(rel_embedding * relation_type_vectors.squeeze(0))
        rel_embedding = torch.stack(emb).to(device)
        # Reshape dimensions.
        r = self.rel_r_linear(rel_embedding)
        i = self.rel_i_linear(rel_embedding)
        j = self.rel_j_linear(rel_embedding)
        k = self.rel_k_linear(rel_embedding)
        dual = self.rel_dual_linear(rel_embedding).chunk(4, dim=-1)
        return (r, i, j, k), dual

    def forward(self, h_idx, r_idx, t_idx):
        # Encode entities and relations.
        h = self.encode_entity(h_idx, entid2name, entid2duc, id2time, id2location)
        t = self.encode_entity(t_idx, entid2name, entid2duc, id2time, id2location)
        r, r_dual = self.encode_relation(r_idx, relid2name)
        # Normalize relations to retain rotation direction.
        r_norm = torch.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2 + r[3] ** 2 + 1e-9)
        r_unit = (r[0] / r_norm, r[1] / r_norm, r[2] / r_norm, r[3] / r_norm)

        # Quaternion multiplication: head entity times unit relation.
        h_r = self.quat_multiply(h, r_unit)

        # Apply magnitude scaling.
        rel_mod = self.rel_mod(r_idx)  # shape: [B, 1]
        h_r_scaled = (h_r[0] * rel_mod, h_r[1] * rel_mod, h_r[2] * rel_mod, h_r[3] * rel_mod)

        # Subtract the tail entity.
        diff = (
            h_r_scaled[0] + r_dual[0] - t[0],
            h_r_scaled[1] + r_dual[1] - t[1],
            h_r_scaled[2] + r_dual[2] - t[2],
            h_r_scaled[3] + r_dual[3] - t[3])

        # Concatenate components and compute the L1/L2 norm.
        diff_concat = torch.cat(diff, dim=1)  # [B, 4*dim]
        score = torch.norm(diff_concat, p=self.p_norm, dim=1)  # [B]

        return score

    def loss_fn(self, pos_score, neg_score):
        # Use the standard margin ranking loss.
        return torch.mean(torch.relu(self.margin - (neg_score - pos_score)))


# ------------------ Training and evaluation ------------------
def generate_negative_batch(batch, num_entities):
    return [(random.randint(0, num_entities - 1), r, t) if random.random() < 0.5 else
            (h, r, random.randint(0, num_entities - 1)) for h, r, t in batch]


def evaluate_loss_on_validation(model, val_data, num_entities, batch_size=128):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for i in range(0, len(val_data), batch_size):
            pos_batch = val_data[i:i + batch_size]
            if not pos_batch:  # Skip empty batches.
                continue

            neg_batch = generate_negative_batch(pos_batch, num_entities)
            pos = torch.LongTensor(pos_batch).T.to(device)
            neg = torch.LongTensor(neg_batch).T.to(device)
            pos_score = model(*pos)
            neg_score = model(*neg)
            loss = model.loss_fn(pos_score, neg_score)
            total_loss += loss.item()
            num_batches += 1

    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


def train(model, optimizer, train_data, val_data, num_entities, batch_size=128, epochs=1000,
          save_path=os.path.join(out_dir, f"best_LLM_LanSE_{os.path.basename(out_dir)}.pt"), patience=30):
    model.to(device)
    best_val_loss = float("inf")
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(1, epochs + 1):
        if early_stop:
            logging.info(f"Early stopping at epoch {epoch}")
            break

        random.shuffle(train_data)
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            pos_batch = train_data[i:i + batch_size]
            if not pos_batch:  # Skip empty batches.
                continue

            neg_batch = generate_negative_batch(pos_batch, num_entities)
            pos = torch.LongTensor(pos_batch).T.to(device)
            neg = torch.LongTensor(neg_batch).T.to(device)
            pos_score = model.forward(*pos)
            neg_score = model(*neg)
            loss = model.loss_fn(pos_score, neg_score)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        logging.info(f"Epoch {epoch}: Avg Train Loss = {avg_loss:.4f}")

        val_loss = evaluate_loss_on_validation(model, val_data, num_entities, batch_size)
        logging.info(f"Epoch {epoch}: Avg Val Loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            logging.info(f"✔ Saved best model at epoch {epoch} with val loss {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info("Early stopping triggered.")
                early_stop = True

    # Load the best model.
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
        logging.info(f"Loaded best model from {save_path}")


def split_data(triples, train_ratio=0.9, val_ratio=0.09):
    random.shuffle(triples)
    n = len(triples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return triples[:train_end], triples[train_end:val_end], triples[val_end:]


def evaluate_hit_at_k_batch_optimized(model, test_triples, ent_total, rel_total, all_true_triples,
                                      batch_size=64, k_list=[1, 3, 10]):
    model.eval()
    hits = {k: 0 for k in k_list}
    ranks = []

    with torch.no_grad():
        for i in range(0, len(test_triples), batch_size):
            batch = test_triples[i:i + batch_size]
            if not batch:  # Skip empty batches.
                continue

            batch_size_actual = len(batch)
            h_ids = torch.tensor([t[0] for t in batch], device=device)
            r_ids = torch.tensor([t[1] for t in batch], device=device)
            t_ids = torch.tensor([t[2] for t in batch], device=device)

            # Create all tail candidates: (batch_size, ent_total).
            all_tails = torch.arange(ent_total, device=device).repeat(batch_size_actual, 1)  # [B, E]

            # Expand heads and relations to match all tail candidates.
            h_ids_expanded = h_ids.unsqueeze(1).repeat(1, ent_total)  # [B, E]
            r_ids_expanded = r_ids.unsqueeze(1).repeat(1, ent_total)  # [B, E]

            # Score each candidate triple.
            scores = []
            for b in range(batch_size_actual):
                # Select batch heads, relations, and candidate tails.
                h_batch = h_ids_expanded[b].unsqueeze(1)  # [E, 1]
                r_batch = r_ids_expanded[b].unsqueeze(1)  # [E, 1]
                t_batch = all_tails[b].unsqueeze(1)  # [E, 1]

                # Reshape to match model inputs.
                h_input = h_batch.squeeze(1)  # [E]
                r_input = r_batch.squeeze(1)  # [E]
                t_input = t_batch.squeeze(1)  # [E]

                # Compute candidate triple scores.
                batch_scores = []
                for e in range(ent_total):
                    score = model(h_input[e:e + 1], r_input[e:e + 1], t_input[e:e + 1])
                    batch_scores.append(score)

                scores.append(torch.cat(batch_scores, dim=0))  # [E]

            scores = torch.stack(scores, dim=0)  # [B, E]

            # Filter known triples.
            filter_mask = torch.zeros_like(scores, dtype=torch.bool)
            for b, (h, r, t) in enumerate(batch):
                for t_cand in range(ent_total):
                    if (h, r, t_cand) in all_true_triples and t_cand != t:
                        filter_mask[b, t_cand] = True
            scores.masked_fill_(filter_mask, float('inf'))

            # Sort candidates.
            _, sorted_idx = torch.sort(scores, dim=1)
            rank = (sorted_idx == t_ids.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1  # [B]
            ranks.extend(rank.cpu().numpy())

            for k in k_list:
                hits[k] += (rank <= k).sum().item()

    if not ranks:  # Handle empty results.
        logging.info("Warning: No ranks calculated")
        return {k: 0.0 for k in k_list}, float('inf'), 0.0

    mr = np.mean(ranks)
    mrr = np.mean(1.0 / np.array(ranks))
    hit_results = {k: hits[k] / len(test_triples) for k in k_list}

    logging.info(f"MR: {mr:.4f}")
    logging.info(f"MRR: {mrr:.4f}")
    for k in k_list:
        logging.info(f"Hit@{k}: {hit_results[k]:.4f}")

    return hit_results, mr, mrr


def precompute_text_embeddings(ent_total, id2ent_name, id2ent_duc, rel_total, id2rel_name, id2time, id2location):
    for i in range(ent_total):
        get_embedding_from_llm(id2ent_name[i])
        get_embedding_from_llm(id2ent_duc[str(i)])
        get_embedding_from_llm(id2time.get(str(i)))
        get_embedding_from_llm(id2location.get(str(i)))
    for idx in range(rel_total):
        get_embedding_from_llm(id2rel_name[idx])


import json

# ------------------ Entry point ------------------
if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser(description='LLM-GeoSE and ablation experiments')
        parser.add_argument('--ablation', default='full', choices=[
            'full', 'wo_entity_enhancement', 'wo_relation_enhancement',
            'wo_text_keywords', 'wo_describe_text', 'wo_time_property',
            'wo_spatial_property', 'wo_time_space'])
        args = parser.parse_args()
        if args.ablation == 'wo_entity_enhancement':
            load_word2vec_resources(use_keywords=False)
        elif args.ablation == 'wo_text_keywords':
            load_word2vec_resources(use_keywords=True)
        ent2id, entid2name = load_mapping("../data/entity2id.txt")
        with open("../data/entityid2duc.json", 'r', encoding='utf-8') as f:
            entid2duc = json.load(f)
        rel2id, relid2name = load_mapping("../data/rel2id.txt")
        with open("../data/id2time.json", 'r', encoding='utf-8') as f:
            id2time = json.load(f)
        with open("../data/id2location.json", 'r', encoding='utf-8') as f:
            id2location = json.load(f)
        with open("../data/relid2typeid.json", 'r', encoding='utf-8') as f:
            relid2typeid = json.load(f)
        triples = load_triples("../data/landslide.txt", ent2id, rel2id)
        train_data = load_triples("../data/train.txt", ent2id, rel2id)
        val_data = load_triples("../data/valid.txt", ent2id, rel2id)
        test_data = load_triples("../data/test.txt", ent2id, rel2id)
        all_true_triples = set(train_data + val_data + test_data)

        model = LLM_LanSE(
            num_entities=len(ent2id),
            num_relations=len(rel2id),
            num_relation_types=4,
            dim=100,
            p_norm=1,
            margin=1,
            ablation=args.ablation
        )
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        epochs = 1000
        batch_size = 2048
        run_out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'training_results', 'LLM-GeoSE', args.ablation)
        os.makedirs(run_out_dir, exist_ok=True)
        train(model, optimizer, train_data, val_data, num_entities=len(ent2id),
              batch_size=batch_size, epochs=epochs,
              save_path=os.path.join(run_out_dir, 'best_LLM-GeoSE.pt'), patience=30)

        logging.info("=== Testing on Test Set ===")
        precompute_text_embeddings(len(ent2id), entid2name, entid2duc, len(rel2id), relid2name, id2time, id2location)
        hits, mr, mrr = evaluate_hit_at_k_batch_optimized(
            model, test_data, len(ent2id), len(rel2id), all_true_triples, batch_size=64
        )
        save_metrics(run_out_dir, 'LLM-GeoSE', hits, mr, mrr)



    finally:
        # Release resources.
        if 'llm' in globals() and llm is not None:
            try:
                del llm
            except:
                pass
