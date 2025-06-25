#!/usr/bin/env python
# calc_token_probs_mlm_v2.py
# ------------------------------------------------------------
"""
BERT 系 (AutoModelForMaskedLM) で
P(candidate | prompt_with_[MASK]) を JSONL 入出力で計算するスクリプト。

改良点
------
* --device に cpu / cuda / cuda:<index> を許容
* GPU 初期化失敗時は自動で CPU にフォールバック
* 入力行数を先読みして tqdm(total=) を設定
* 10 行ごとに flush + fsync で書き込みを確定
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer


# ------------------------------------------------------------
# 1. 1 プロンプト分の確率計算
# ------------------------------------------------------------
@torch.no_grad()
def calc_probs_one_mlm(
    model,
    tokenizer,
    prompt_template: str,
    candidates: List[str],
    device: torch.device,
) -> List[float]:
    """
    Return P(c | prompt_template) for each c in candidates (order preserved).

    prompt_template には最低 1 個の [MASK] が必要。
    候補が N サブトークンに分割される場合、その [MASK] を N 個に自動展開して計算。
    """
    mask_token = tokenizer.mask_token
    if mask_token not in prompt_template:
        raise ValueError(f"Prompt must contain '{mask_token}'")

    probs: List[float] = []
    for cand in candidates:
        cand_ids = tokenizer(cand, add_special_tokens=False).input_ids
        num_mask = len(cand_ids)

        # [MASK] を必要数に置換（最初の 1 か所だけ置換）
        masked_seq = " ".join([mask_token] * num_mask)
        prompt = prompt_template.replace(mask_token, masked_seq, 1)

        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        logits = model(**enc).logits            # (1, seq_len, vocab)
        log_probs = F.log_softmax(logits, dim=-1)

        # [MASK] の位置を取得
        mask_pos = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero(as_tuple=False).squeeze(-1)

        # log P の和を取る
        total_log_p = torch.tensor(0.0, device=device)
        for idx, pos in enumerate(mask_pos):
            total_log_p += log_probs[0, pos, cand_ids[idx]]

        probs.append(torch.exp(total_log_p).item())

    return probs


# ------------------------------------------------------------
# 2. メイン
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="モデル名またはローカルパス")
    parser.add_argument("--input", required=True, help="入力 JSONL")
    parser.add_argument("--output", required=True, help="出力 JSONL")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu / cuda / cuda:<index> (デフォルト: 自動判定)",
    )
    args = parser.parse_args()

    # --- デバイス設定 & モデル読込 ---
    try:
        device = torch.device(args.device)
    except AssertionError:
        raise ValueError(f"--device に無効な値が渡されました: {args.device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForMaskedLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16 if device.type == "cuda" else None,
        ).to(device)
    except (RuntimeError, AssertionError) as e:
        # GPU 関連のエラーは CPU にフォールバック
        print(f"[WARN] CUDA init failed ({e}); falling back to CPU")
        device = torch.device("cpu")
        model = AutoModelForMaskedLM.from_pretrained(args.model).to(device)
    model.eval()

    in_path, out_path = Path(args.input), Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 行数を数えてプログレスバーに渡す ---
    total_lines = sum(1 for line in in_path.open(encoding="utf-8") if line.strip())

    with in_path.open(encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for i, line in enumerate(tqdm(fin, desc="processing", total=total_lines, ncols=80)):
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["prob"] = calc_probs_one_mlm(
                model,
                tokenizer,
                obj["prompt"],      # [MASK] を含む
                obj["candidates"],
                device=device,
            )
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

            # 10 行ごとに確定保存
            if (i + 1) % 10 == 0:
                fout.flush()
                os.fsync(fout.fileno())

        # 端数分 flush
        fout.flush()
        os.fsync(fout.fileno())


if __name__ == "__main__":
    main()