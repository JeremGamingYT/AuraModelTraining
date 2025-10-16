#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entraînement séquence-niveau (PPO) sur TPU v5e-8 pour QA factuelle courte.
- Politique: T5ForConditionalGeneration (pré-entraîné en span corruption, donc pas du next-token unidirectionnel)
- Récompense: Exact Match + F1 (token), pénalité de longueur, + KL vers une policy de référence gelée
- Baseline (Value): tête linéaire sur l'encodeur (mean pooling) -> V(x) (scalaire)
- Optim: PPO (clipped) + MSE pour la value

Données: CSV avec colonnes: prompt,answer

Lancer (sur TPU VM, 8 cœurs):
  python3 -m torch_xla.distributed.xla_spawn --num_processes 8 train_seqppo_tpu.py \
    --train_csv /path/data/train.csv \
    --eval_csv /path/data/dev.csv \
    --model_name t5-small \
    --output_dir /path/ckpts/seqppo_t5small \
    --per_device_batch_size 4 \
    --ppo_epochs 4 \
    --steps 5000

Conseils TPU:
  export PJRT_DEVICE=TPU
  export XLA_USE_BF16=1
  # Optionnel, mais recommandé sur v5e:
  export XLA_IR_DEBUG=0 XLA_HLO_DEBUG=0

Dépendances (sur TPU VM):
  pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
  pip install torch-xla==2.3 -f https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.3-cp310-cp310-linux_x86_64.whl
  pip install transformers datasets accelerate sentencepiece pandas numpy

Remarque: Génération = sampling (exploration); pas de MLE/next-token dans la loss.
"""

import os
# Définir par défaut pour PJRT si pas déjà défini.
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ.setdefault("XLA_USE_BF16", "1")

import math
import argparse
import random
import unicodedata
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler

# PyTorch/XLA
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

# HF
from transformers import T5ForConditionalGeneration, T5TokenizerFast, AutoConfig, AutoTokenizer


# =========================
# Utils: seed & normalisation texte
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    # minuscules, strip, enlever accents, espaces multiples
    s = s.strip().lower()
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = ' '.join(s.split())
    return s

def em_reward(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0

def f1_reward(pred: str, gold: str) -> float:
    p_tokens = normalize_text(pred).split()
    g_tokens = normalize_text(gold).split()
    if len(p_tokens) == 0 and len(g_tokens) == 0:
        return 1.0
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return 0.0
    common = 0
    g_counts = {}
    for t in g_tokens:
        g_counts[t] = g_counts.get(t, 0) + 1
    for t in p_tokens:
        if g_counts.get(t, 0) > 0:
            common += 1
            g_counts[t] -= 1
    if common == 0:
        return 0.0
    precision = common / len(p_tokens)
    recall = common / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


# =========================
# Dataset
# =========================

class QADataset(Dataset):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        assert "prompt" in df.columns and "answer" in df.columns, "CSV doit contenir 'prompt' et 'answer'."
        self.samples = df[["prompt", "answer"]].astype(str).to_dict(orient="records")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


@dataclass
class Collator:
    tokenizer: T5TokenizerFast
    max_input_len: int = 128

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, Any]:
        prompts = [ex["prompt"] for ex in batch]
        answers = [ex["answer"] for ex in batch]
        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "prompts": prompts,
            "answers": answers,
        }


# =========================
# Value Head (baseline V(x))
# =========================

class EncoderValueHead(nn.Module):
    """
    Prend l'encoder T5 -> mean-pool -> linéaire -> scalaire.
    On passe les 'inputs' du même tokenizer.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, encoder_last_hidden_state, attention_mask):
        # encoder_last_hidden_state: [B, S, H]
        mask = attention_mask.unsqueeze(-1)  # [B, S, 1]
        masked = encoder_last_hidden_state * mask
        lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)  # [B,1]
        mean_pool = masked.sum(dim=1) / lengths  # [B, H]
        v = self.value(mean_pool).squeeze(-1)  # [B]
        return v


# =========================
# Logprob util
# =========================

def token_logprobs_from_labels(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    logits: [B, T, V]
    labels: [B, T]  (avec -100 pour ignorer)
    retourne: logp_sum par séquence [B]
    """
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, V]
    labels_ = labels.clone()
    mask = (labels_ != -100)
    labels_[~mask] = 0  # pour gather sans OOB
    gathered = log_probs.gather(dim=-1, index=labels_.unsqueeze(-1)).squeeze(-1)  # [B, T]
    logp_per_seq = (gathered * mask).sum(dim=-1)  # [B]
    return logp_per_seq


def build_labels_from_generated(generated_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """
    Convertit les ids générés en labels compatibles T5 (pad -> -100).
    """
    labels = generated_ids.clone()
    labels[labels == pad_token_id] = -100
    return labels


# =========================
# PPO components
# =========================

@dataclass
class PPOConfig:
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0  # T5 a déjà une bonne entropie avec sampling; laisser à 0
    kl_coef: float = 0.01      # coefficient pour KL (reward shaping)
    length_penalty: float = 0.0
    ppo_epochs: int = 4
    lr: float = 1e-5
    lr_value: float = 1e-4
    max_new_tokens: int = 16
    top_p: float = 0.9
    top_k: int = 50
    temperature: float = 1.0


# =========================
# Entraînement par process (pour xmp.spawn)
# =========================

def train_loop(index: int, args):
    # Device XLA
    device = xm.xla_device()

    # Seed par-rang
    set_seed(args.seed + xm.get_ordinal())

    # Tokenizer & modèles
    tokenizer = T5TokenizerFast.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(args.model_name)
    policy = T5ForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16 if os.environ.get("XLA_USE_BF16","0")=="1" else None)
    ref_policy = T5ForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16 if os.environ.get("XLA_USE_BF16","0")=="1" else None)

    # Geler la policy de référence
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    # Value head (sur l'encodeur)
    value_head = EncoderValueHead(hidden_size=policy.config.d_model)

    # To device
    policy.to(device)
    ref_policy.to(device)
    value_head.to(device)

    policy.train()
    ref_policy.eval()
    value_head.train()

    # Datasets & Loaders
    train_ds = QADataset(args.train_csv)
    eval_ds = QADataset(args.eval_csv) if args.eval_csv else None

    collator = Collator(tokenizer=tokenizer, max_input_len=args.max_input_len)

    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
        drop_last=True
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=False,
    )

    if eval_ds:
        eval_sampler = DistributedSampler(
            eval_ds,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False,
            drop_last=False
        )
        eval_loader = DataLoader(
            eval_ds,
            batch_size=args.per_device_batch_size,
            sampler=eval_sampler,
            collate_fn=collator,
            num_workers=0,
            pin_memory=False,
        )
    else:
        eval_loader = None

    # Optimizers
    ppo_conf = PPOConfig(
        clip_range=args.clip_range,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        kl_coef=args.kl_coef,
        length_penalty=args.length_penalty,
        ppo_epochs=args.ppo_epochs,
        lr=args.lr,
        lr_value=args.lr_value,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        temperature=args.temperature,
    )

    policy_opt = torch.optim.AdamW(policy.parameters(), lr=ppo_conf.lr)
    value_opt = torch.optim.AdamW(value_head.parameters(), lr=ppo_conf.lr_value)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    decoder_start = policy.config.decoder_start_token_id

    # Step counter partagé (juste indicatif)
    global_step = 0

    xm.master_print("Démarrage entraînement PPO séquence-niveau…")

    # Boucle d'entraînement
    while global_step < args.steps:
        train_sampler.set_epoch(global_step)  # pour reshuffle
        for batch in train_loader:
            if global_step >= args.steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            gold_answers: List[str] = batch["answers"]

            # ----- Rollout: Génération avec la policy (sans gradient)
            with torch.no_grad():
                gen_out = policy.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    top_p=ppo_conf.top_p,
                    top_k=ppo_conf.top_k,
                    temperature=ppo_conf.temperature,
                    max_new_tokens=ppo_conf.max_new_tokens,
                    num_return_sequences=1,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )
                # Les sorties sont uniquement la partie générée. Pour obtenir logprobs,
                # on recalcule avec labels = gen_out (teacher-forcing sur la réponse).
                labels = build_labels_from_generated(gen_out, pad_id)

                # Logprobs de la policy (ancienne) pour PPO ratio + KL w.r.t ref
                out_policy = policy(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                logp_old = token_logprobs_from_labels(out_policy.logits, labels, pad_id)  # [B]

                out_ref = ref_policy(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                logp_ref = token_logprobs_from_labels(out_ref.logits, labels, pad_id)  # [B]

                # KL approx: sum(log pi - log pref)
                kl_old = (logp_old - logp_ref)  # [B]

                # Texte décodé pour reward EM/F1
                pred_texts = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
                # Récompenses
                rewards = []
                for pred, gold in zip(pred_texts, gold_answers):
                    r = 0.0
                    r += 0.7 * em_reward(pred, gold)
                    r += 0.3 * f1_reward(pred, gold)
                    if ppo_conf.length_penalty > 0.0:
                        r -= ppo_conf.length_penalty * max(0, len(pred.split()) - 5)  # petite pénalité si long
                    rewards.append(r)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

                # Reward shaping avec KL (garder proche du ref)
                shaped_rewards = rewards - ppo_conf.kl_coef * kl_old

                # Baseline V(x)
                enc_outputs = policy.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
                values = value_head(enc_outputs.last_hidden_state, attention_mask)  # [B]

                # Avantage (une étape)
                advantages = (shaped_rewards - values).detach()

                # On stocke les éléments du rollout pour PPO update
                rollout = {
                    "input_ids": input_ids.detach(),
                    "attention_mask": attention_mask.detach(),
                    "labels": labels.detach(),
                    "old_logp": logp_old.detach(),
                    "advantages": advantages.detach(),
                    "returns": shaped_rewards.detach()
                }

            # ----- PPO update: plusieurs epochs sur ce rollout
            for _ in range(ppo_conf.ppo_epochs):
                # Recompute logp_new
                out_new = policy(
                    input_ids=rollout["input_ids"],
                    attention_mask=rollout["attention_mask"],
                    labels=rollout["labels"]
                )
                logp_new = token_logprobs_from_labels(out_new.logits, rollout["labels"], pad_id)  # [B]

                ratio = torch.exp(logp_new - rollout["old_logp"])  # [B]
                unclipped = ratio * rollout["advantages"]
                clipped = torch.clamp(ratio, 1.0 - ppo_conf.clip_range, 1.0 + ppo_conf.clip_range) * rollout["advantages"]
                policy_loss = -torch.mean(torch.min(unclipped, clipped))

                # (Optionnel) Entropie - ici non utilisée (déjà sampling)
                # entropy = - (torch.exp(logp_new) * logp_new) ... pas trivial sur sum-logp; on skip

                # Value loss
                with torch.no_grad():
                    enc_outputs = policy.get_encoder()(
                        input_ids=rollout["input_ids"],
                        attention_mask=rollout["attention_mask"]
                    )
                values_new = value_head(enc_outputs.last_hidden_state, rollout["attention_mask"])
                value_loss = torch.mean((values_new - rollout["returns"]) ** 2)

                loss = policy_loss + ppo_conf.value_coef * value_loss

                policy_opt.zero_grad(set_to_none=True)
                value_opt.zero_grad(set_to_none=True)
                loss.backward()

                # Step XLA
                xm.optimizer_step(policy_opt, barrier=True)
                xm.optimizer_step(value_opt, barrier=False)

            # ----- Logging simple (master only)
            if global_step % args.log_every == 0 and xm.is_master_ordinal():
                # calcule quelques métriques locales
                with torch.no_grad():
                    avg_reward = rewards.mean().item()
                    avg_em = float(np.mean([em_reward(p, g) for p, g in zip(pred_texts, gold_answers)]))
                    avg_f1 = float(np.mean([f1_reward(p, g) for p, g in zip(pred_texts, gold_answers)]))
                xm.master_print(
                    f"[step {global_step}] reward={avg_reward:.3f} em={avg_em:.3f} f1={avg_f1:.3f} "
                    f"loss={loss.item():.3f} polyloss={policy_loss.item():.3f} vloss={value_loss.item():.3f}"
                )

            # ----- Éval rapide (optionnelle)
            if eval_loader and (global_step % args.eval_every == 0) and xm.is_master_ordinal():
                policy.eval()
                with torch.no_grad():
                    ems, f1s = [], []
                    neval = 0
                    for evb in eval_loader:
                        ev_in = evb["input_ids"].to(device)
                        ev_am = evb["attention_mask"].to(device)
                        ev_ans = evb["answers"]
                        ev_gen = policy.generate(
                            input_ids=ev_in,
                            attention_mask=ev_am,
                            do_sample=False,
                            max_new_tokens=ppo_conf.max_new_tokens,
                            pad_token_id=pad_id,
                            eos_token_id=eos_id,
                        )
                        ev_pred = tokenizer.batch_decode(ev_gen, skip_special_tokens=True)
                        for p, g in zip(ev_pred, ev_ans):
                            ems.append(em_reward(p, g))
                            f1s.append(f1_reward(p, g))
                            neval += 1
                        if neval >= args.max_eval_samples:
                            break
                    xm.master_print(f"[eval @ {global_step}] EM={np.mean(ems):.3f} F1={np.mean(f1s):.3f}")
                policy.train()

            # ----- Checkpoint (master only)
            if (global_step % args.ckpt_every == 0 or global_step == args.steps - 1) and xm.is_master_ordinal():
                os.makedirs(args.output_dir, exist_ok=True)
                xm.master_print(f"Saving checkpoint @ step {global_step} -> {args.output_dir}")
                # Sauvegarder la policy & tokenizer (standard HF)
                policy.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                # Value head (poids séparés)
                torch.save(value_head.state_dict(), os.path.join(args.output_dir, "value_head.pt"))

            global_step += 1

    xm.master_print("✅ Entraînement terminé.")


# =========================
# Argparse & main
# =========================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", type=str, required=True, help="CSV avec colonnes prompt,answer")
    p.add_argument("--eval_csv", type=str, default=None)
    p.add_argument("--model_name", type=str, default="t5-small")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--per_device_batch_size", type=int, default=4)
    p.add_argument("--max_input_len", type=int, default=128)

    # PPO & génération
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_value", type=float, default=1e-4)
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--kl_coef", type=float, default=0.01)
    p.add_argument("--length_penalty", type=float, default=0.0)

    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)

    # Logs & eval & ckpt
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--max_eval_samples", type=int, default=512)
    p.add_argument("--ckpt_every", type=int, default=500)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    # spawn 8 workers sur TPU v5e-8
    xmp.spawn(train_loop, args=(args,), nprocs=8, start_method="fork")


if __name__ == "__main__":
    main()