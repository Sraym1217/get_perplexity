import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import openai
import tiktoken           # OpenAI の公式トークナイザ
from tqdm import tqdm


# ---------------------- 1. 共通ユーティリティ ---------------------- #
def _get_token_logprobs(
    model: str,
    text: str,
    max_logprobs: int = 100,
) -> Tuple[List[int], List[float]]:
    """
    text 全体を 1 つの prompt として `echo=True` で呼び出し、
    各トークンの (token_id, logprob) ペアを返す。
    """
    resp = openai.Completion.create(
        model=model,
        prompt=text,
        max_tokens=0,       # 生成しない
        echo=True,
        logprobs=max_logprobs,
    )
    # `resp['choices'][0]['logprobs']` に tokens と logprobs が並ぶ
    tokens      = resp["choices"][0]["logprobs"]["tokens"]
    token_ids   = resp["choices"][0]["logprobs"]["token_ids"]
    token_probs = resp["choices"][0]["logprobs"]["token_logprobs"]
    return token_ids, token_probs


def _calc_prob_one(
    model: str,
    enc,
    prompt: str,
    candidates: List[str],
) -> List[float]:
    """
    P(c | prompt) を候補ごとに計算。
    """
    probs: List[float] = []

    # prompt のトークン列
    prompt_ids = enc.encode(prompt, disallowed_special=())

    for cand in candidates:
        cand_ids = enc.encode(cand, disallowed_special=())

        # ---- (A) 単一トークン ----
        if len(cand_ids) == 1:
            # prompt 部分だけ送り、logprobs で次トークン分布を得る方法もあるが
            # Instruct 系モデルは `echo=True` がシンプル
            _, token_probs = _get_token_logprobs(model, prompt, max_logprobs=100)

            next_token_probs = token_probs[-1]   # prompt 末尾の次トークン分布
            # token_probs[-1] は dict ではなく個々の logprob なので
            # `logprobs=k` の場合は top-k しか来ない点に注意
            # → 一般には cand が top-k に必ず含まれるとは限らない。
            #    ここでは fallback として 0 にする。
            prob = 0.0
            if cand_ids[0] in next_token_probs:
                prob = next_token_probs[cand_ids[0]]
            probs.append(prob)
            continue

        # ---- (B) 複数トークン ----
        # 「prompt + candidate」丸ごと echo=True で呼び、
        # candidate 部の logprob を累積する
        full_text = prompt + cand
        full_ids, full_logprobs = _get_token_logprobs(model, full_text)

        # prompt 部分をスキップして candidate の logprob を合算
        cand_start = len(prompt_ids)
        log_p = sum(full_logprobs[cand_start : cand_start + len(cand_ids)])

        probs.append(pow(2.718281828459045, log_p))  # exp(log_p)

    return probs


# ---------------------- 2. メイン ---------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="例: gpt-3.5-turbo-instruct")
    parser.add_argument("--input", required=True, help="入力 JSONL")
    parser.add_argument("--output", required=True, help="出力 JSONL")
    args = parser.parse_args()

    # --- tiktoken エンコーダを取得 ---
    enc = tiktoken.encoding_for_model(args.model)

    in_path, out_path = Path(args.input), Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_lines = sum(1 for line in in_path.open(encoding="utf-8") if line.strip())

    with in_path.open(encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        line_idx = 0
        for line in tqdm(fin, total=total_lines, desc="processing", ncols=80):
            if not line.strip():
                continue
            obj = json.loads(line)
            obj["prob"] = _calc_prob_one(
                args.model, enc, obj["prompt"], obj["candidates"]
            )

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            line_idx += 1

            # 10 件ごとに flush & fsync
            if line_idx % 10 == 0:
                fout.flush()
                os.fsync(fout.fileno())

        fout.flush()
        os.fsync(fout.fileno())


if __name__ == "__main__":
    main()