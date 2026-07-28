#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""JSON database helpers for the Iris Recognition System web app.

Python 3.6 compatible.
"""

import json
import os
import time

import numpy as np


DB_VERSION = 2
MIN_EMBEDDING_NORM = 1e-6


def now_text():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def empty_db():
    return {"version": DB_VERSION, "users": {}}


def ensure_parent_dir(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent)


def load_db(path):
    if not os.path.exists(path):
        return empty_db()

    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except Exception:
        return empty_db()

    if isinstance(raw, dict) and isinstance(raw.get("users"), dict):
        if "version" not in raw:
            raw["version"] = DB_VERSION
        return raw

    # Migrate old format: {"Dat": [embedding...]}
    db = empty_db()
    if isinstance(raw, dict):
        for name, emb in raw.items():
            if isinstance(emb, list):
                db["users"][str(name)] = {
                    "created_at": "legacy",
                    "left": emb,
                    "right": None,
                    "notes": "Migrated from old single-eye DB"
                }
    return db


def save_db(path, db):
    ensure_parent_dir(path)
    with open(path, "w") as f:
        json.dump(db, f, indent=2)


def reset_db(path):
    save_db(path, empty_db())


def normalize_embedding(embedding):
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        raise RuntimeError("Embedding is empty")
    if not np.all(np.isfinite(arr)):
        raise RuntimeError("Embedding contains invalid numbers")

    norm = float(np.linalg.norm(arr))
    if (not np.isfinite(norm)) or norm < MIN_EMBEDDING_NORM:
        raise RuntimeError("Embedding norm is zero or invalid")
    return arr / norm


def cosine(a, b):
    aa = normalize_embedding(a)
    bb = normalize_embedding(b)
    return float(np.dot(aa, bb))


def db_summary(path):
    db = load_db(path)
    users = sorted(list(db.get("users", {}).keys()))
    return {"count": len(users), "users": users}


def list_users(path):
    db = load_db(path)
    users = []
    for name, rec in sorted(db.get("users", {}).items()):
        if not isinstance(rec, dict):
            continue
        users.append({
            "name": name,
            "created_at": rec.get("created_at", "unknown"),
            "left": rec.get("left") is not None,
            "right": rec.get("right") is not None,
            "notes": rec.get("notes", "")
        })
    return users


def register_both(path, name, left_emb, right_emb):
    clean_name = (name or "").strip()
    if not clean_name:
        raise RuntimeError("Name is required")

    left = normalize_embedding(left_emb).tolist()
    right = normalize_embedding(right_emb).tolist()

    db = load_db(path)
    db["users"][clean_name] = {
        "created_at": now_text(),
        "left": left,
        "right": right,
        "notes": "Two-eye enrollment from web UI"
    }
    save_db(path, db)
    return db_summary(path)


def clean_eye_filter(eye_filter):
    value = str(eye_filter or "").strip().lower()
    if value in ("l", "left"):
        return "left"
    if value in ("r", "right"):
        return "right"
    return None


def recognize_with_options(path, embedding, threshold, top_k, eye_filter=None, margin=0.0):
    query = normalize_embedding(embedding)
    db = load_db(path)
    users = db.get("users", {})
    if not users:
        return {
            "matched": False,
            "name": None,
            "eye": None,
            "score": None,
            "top_scores": [],
            "second_best_different_identity_score": None,
            "score_margin": None,
            "threshold_pass": False,
            "margin_pass": True,
            "eye_filter": clean_eye_filter(eye_filter),
            "required_margin": float(margin or 0.0)
        }

    best_name = None
    best_eye = None
    best_score = -999.0
    top = []
    eye_filter = clean_eye_filter(eye_filter)
    eyes = [eye_filter] if eye_filter else ["left", "right"]
    required_margin = float(margin or 0.0)

    for name, rec in users.items():
        if not isinstance(rec, dict):
            continue
        for eye in eyes:
            ref = rec.get(eye)
            if ref is None:
                continue
            try:
                score = cosine(query, ref)
                bad = False
            except Exception:
                score = 0.0
                bad = True

            top.append({
                "user": name,
                "eye": eye,
                "score": score,
                "bad_embedding": bad
            })
            if (not bad) and score > best_score:
                best_name = name
                best_eye = eye
                best_score = score

    top.sort(key=lambda x: x["score"], reverse=True)
    if best_score <= -998.0:
        return {
            "matched": False,
            "name": None,
            "eye": None,
            "score": 0.0,
            "top_scores": top[:top_k],
            "second_best_different_identity_score": None,
            "score_margin": None,
            "threshold_pass": False,
            "margin_pass": True,
            "eye_filter": eye_filter,
            "required_margin": required_margin
        }

    second_diff = None
    for item in top:
        if item.get("bad_embedding"):
            continue
        if item.get("user") != best_name:
            second_diff = float(item.get("score", 0.0))
            break

    if second_diff is None:
        score_margin = None
        margin_pass = True
    else:
        score_margin = best_score - second_diff
        margin_pass = required_margin <= 0.0 or score_margin >= required_margin

    threshold_pass = best_score >= threshold
    matched = threshold_pass and margin_pass
    return {
        "matched": matched,
        "name": best_name if matched else None,
        "eye": best_eye,
        "score": best_score,
        "top_scores": top[:top_k],
        "second_best_different_identity_score": second_diff,
        "score_margin": score_margin,
        "threshold_pass": threshold_pass,
        "margin_pass": margin_pass,
        "eye_filter": eye_filter,
        "required_margin": required_margin
    }


def recognize(path, embedding, threshold, top_k):
    result = recognize_with_options(path, embedding, threshold, top_k)
    return result["name"], result["eye"], result["score"], result["top_scores"]
