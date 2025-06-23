#!/usr/bin/env python
# coding: utf-8
"""
batch_prob_calc.py
------------------
LLM で P(candidate | prompt) を行単位で計算し、JSONL で出力します。
 * tqdm で全体進捗を表示
 * 10 行ごとにファイルを確定保存
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------------
# 1. 1 行 (=1 プロンプト) 分の確率計算
# ------------------------------------------------------------
def calc_probs_one(
    model,
    tokenizer,
    prompt: str,
    candidates: List[str],
    device: str = "cpu",
) -> List[float]:
    """P(c | prompt) for each c in candidates, preserving order."""
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # prompt 末尾の次トークン確率
    with torch.no_grad():
        base_logits = model(prompt_ids).logits
        base_probs = F.softmax(base_logits[0, -1], dim=-1)

    probs: List[float] = []
    for cand in candidates:
        cand_ids = tokenizer(cand, add_special_tokens=False).input_ids

        # 単一トークン
        if len(cand_ids) == 1:
            probs.append(base_probs[cand_ids[0]].item())
            continue

        # 複数トークン
        log_p = torch.log(base_probs[cand_ids[0]])
        ctx_ids = torch.cat(
            [prompt_ids, torch.tensor([[cand_ids[0]]], device=device)], dim=-1
        )

        for tid in cand_ids[1:]:
            with torch.no_grad():
                step_logits = model(ctx_ids).logits
            step_probs = F.softmax(step_logits[0, -1], dim=-1)
            log_p += torch.log(step_probs[tid])
            ctx_ids = torch.cat([ctx_ids, torch.tensor([[tid]], device=device)], dim=-1)

        probs.append(torch.exp(log_p).item())

    return probs


# ------------------------------------------------------------
# 2. メイン
# ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="モデル名 or ローカルディレクトリ")
    parser.add_argument("--input", required=True, help="入力 JSONL ファイル")
    parser.add_argument("--output", required=True, help="出力 JSONL ファイル")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="推論デバイス",
    )
    args = parser.parse_args()

    # --- モデル / トークナイザ読込 ---
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else None,
    ).to(args.device)
    model.eval()

    in_path, out_path = Path(args.input), Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 総行数を先読みしてプログレスバー設定 ---
    total_lines = sum(1 for line in in_path.open(encoding="utf-8") if line.strip())

    with in_path.open(encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        line_idx = 0  # 行番号（0 始まり）
        for line in tqdm(fin, desc="processing", total=total_lines, ncols=80):
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["prob"] = calc_probs_one(
                model,
                tokenizer,
                obj["prompt"],
                obj["candidates"],
                device=args.device,
            )

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            line_idx += 1

            # --- 10 行ごとに確定保存 ---
            if line_idx % 10 == 0:
                fout.flush()           # Python バッファ → OS バッファ
                os.fsync(fout.fileno())  # OS バッファ → ディスク

        # 端数分を忘れず flush
        fout.flush()
        os.fsync(fout.fileno())


if __name__ == "__main__":
    main()