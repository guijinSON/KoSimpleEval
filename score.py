import os
import sys
import argparse
import pandas as pd
from typing import Dict, Optional, List, Tuple
from math_verify import parse, verify

SEP = "-" * 2  # two hyphens without writing them literally


def safe_name(s: str) -> str:
    return s.replace("/", "_").strip()


def get_model_and_benchmark(filename: str) -> Tuple[str, str]:
    """
    Extract benchmark and model names from a file name using the two-hyphen separator.
    Returns (model_name, benchmark_name).
    """
    base = os.path.basename(filename)
    if not base.endswith(".csv"):
        return "", ""
    stem = base[:-4]
    parts = stem.split(SEP)
    if len(parts) < 2:
        return "", ""
    benchmark = parts[0].strip()
    model = parts[-1].strip()
    return model, benchmark


def drop_think(response: str) -> str:
    if not isinstance(response, str):
        return ""
    tag = "</think>"
    if tag in response:
        return response.split(tag, 1)[1]
    return response


def parse_last_token(text: str):
    try:
        toks = parse(text)
        return toks[-1] if toks else None
    except Exception:
        return None


NUM2LETTER = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
LETTER2NUM = {v: k for k, v in NUM2LETTER.items()}


def to_pred_num(x) -> Optional[int]:
    # Accept 1..5 as int or numeric string
    if isinstance(x, (int, float)):
        try:
            xi = int(x)
            return xi
        except Exception:
            pass
    if isinstance(x, str):
        xs = x.strip()
        if xs.isdigit():
            try:
                return int(xs)
            except Exception:
                pass
        u = xs.upper()
        if u in LETTER2NUM:
            return LETTER2NUM[u]
    return None


def to_gold_num(x) -> Optional[int]:
    # Convert gold to 1..5 if it is either numeric or a letter A..E
    if isinstance(x, (int, float)):
        try:
            return int(x)
        except Exception:
            return None
    if isinstance(x, str):
        xs = x.strip()
        if xs.isdigit():
            try:
                return int(xs)
            except Exception:
                return None
        u = xs.upper()
        if u in LETTER2NUM:
            return LETTER2NUM[u]
    return None


def evaluate_mcqa(df: pd.DataFrame) -> Dict[str, float]:
    """
    Multiple choice evaluator.
    - Extract final prediction token.
    - Map prediction to 1..5 where possible.
    - Compare to numeric gold.
    """
    # Clean the model response and parse last token
    preds: List[Optional[str]] = []
    for _, row in df.iterrows():
        resp_text = drop_think(row.get("response", ""))
        last = parse_last_token(resp_text)
        preds.append(last if last is not None else None)
    df["pred"] = preds

    # Convert pred and gold to numbers
    df["pred_num"] = df["pred"].apply(to_pred_num)
    df["gold_num"] = df["gold"].apply(to_gold_num)

    df["correct"] = (df["pred_num"].notna()) & (df["gold_num"].notna()) & (df["pred_num"] == df["gold_num"])

    return compute_scores(df)


def evaluate_mcqa_raw(df: pd.DataFrame) -> Dict[str, float]:
    """
    Raw string equality for tasks like ClinicalQA, KoBALT-700.
    Compares parsed final token string to the gold string.
    """
    preds: List[Optional[str]] = []
    for _, row in df.iterrows():
        resp_text = drop_think(row.get("response", ""))
        last = parse_last_token(resp_text)
        preds.append(str(last) if last is not None else None)
    df["pred"] = preds

    # Normalize strings lightly for comparison
    def norm(x):
        return str(x).strip()

    df["correct"] = df["pred"].apply(norm) == df["gold"].apply(norm)

    return compute_scores(df)


def evaluate_open_math(df: pd.DataFrame) -> Dict[str, float]:
    """
    Open-ended math evaluator.
    Correct if either:
    1) verify(parse(str(gold)), parse(response)) is True
    2) last parsed token equals the gold as a string
    """
    corrects: List[bool] = []
    for _, row in df.iterrows():
        try:
            resp_text = drop_think(row.get("response", ""))
            resp_tokens = parse(resp_text)
            gold_tokens = parse(str(row.get("gold", "")))
            is_correct0 = verify(gold_tokens, resp_tokens)
            last_tok = resp_tokens[-1] if resp_tokens else None
            is_correct1 = str(row.get("gold", "")) == (str(last_tok) if last_tok is not None else None)
            corrects.append(bool(is_correct0 or is_correct1))
        except Exception:
            corrects.append(False)
    df["correct"] = corrects

    return compute_scores(df)


def compute_scores(df: pd.DataFrame) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if df.empty or "correct" not in df.columns:
        return scores
    scores["overall_accuracy"] = float(df["correct"].mean() * 100.0)
    if "category" in df.columns:
        by_cat = df.groupby("category")["correct"].mean() * 100.0
        for cat, val in by_cat.items():
            scores[f"cat_{cat}"] = float(val)
    return scores


def write_scores_csv(scores: Dict[str, float], out_csv: str) -> None:
    """
    Save a single-row CSV with friendly column names:
    - overall_accuracy -> Average
    - cat_X -> Category_X
    """
    if not scores:
        print("No scores to write.")
        return
    final = {}
    for k, v in scores.items():
        if k == "overall_accuracy":
            final["Average"] = v
        elif k.startswith("cat_"):
            final[f"Category_{k[4:]}"] = v
        else:
            final[k] = v
    df = pd.DataFrame([final])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Wrote {out_csv}")


import os
from typing import Optional, List

def normalize_model_tag(s: Optional[str]) -> str:
    """
    Normalize a model tag so that variations like
    hyphens vs underscores or dots do not block matching.
    """
    if not s:
        return ""
    t = str(s)
    t = t.replace("/", "_")
    t = t.replace("-", "_")
    t = t.replace(".", "")
    return t.lower().strip()

def find_file(source_dir: str, dataset: str, model: Optional[str]) -> Optional[str]:
    """
    Locate a CSV named like:
      <dataset><sep><model>[optional_suffix].csv
    where <sep> can be two hyphens or one hyphen.
    The model is matched exactly if possible, otherwise by normalized prefix.
    """
    # expected sanitized model from the caller
    safe_model = model.replace("/", "_") if model else None
    norm_target = normalize_model_tag(safe_model) if safe_model else ""

    candidates: List[tuple[int, str]] = []  # (score, fullpath)

    try:
        names = os.listdir(source_dir)
    except FileNotFoundError:
        return None

    for name in names:
        if not name.endswith(".csv"):
            continue

        stem = os.path.basename(name)[:-4]

        # try dataset + two hyphens, then dataset + one hyphen
        model_part = None
        for sep in ("-" * 2, "-"):
            prefix = f"{dataset}{sep}"
            if stem.startswith(prefix):
                model_part = stem[len(prefix):]
                break
        if model_part is None:
            continue

        # if a model filter was provided, score the match
        if safe_model:
            exact = (model_part == safe_model)
            norm_model = normalize_model_tag(model_part)
            prefix_match = norm_model.startswith(norm_target)
            contain_match = (norm_target in norm_model)
            if not (exact or prefix_match or contain_match):
                continue
            score = 2 if exact else 1 if prefix_match else 0
        else:
            score = 0

        candidates.append((score, os.path.join(source_dir, name)))

    if not candidates:
        return None

    # prefer exact match, then normalized prefix, then any
    candidates.sort(key=lambda x: (-x[0], x[1]))
    top_score = candidates[0][0]
    best = [p for s, p in candidates if s == top_score]
    if len(best) > 1:
        print(f"Warning: multiple matches found with same score. Using the first.\n{best}")
    return best[0]


def to_presentable_scores(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Map internal keys to the friendly CSV keys you already write:
      overall_accuracy -> Average
      cat_X -> Category_X
    Other keys are passed through unchanged.
    """
    final = {}
    for k, v in scores.items():
        if k == "overall_accuracy":
            final["Average"] = v
        elif k.startswith("cat_"):
            final[f"Category_{k[4:]}"] = v
        else:
            final[k] = v
    return final


def update_dataset_summary(
    target_dir: str,
    safe_dataset: str,
    safe_model: str,
    presentable_scores: Dict[str, float],
    style: str
) -> None:
    """
    Append or upsert one row into
      <target>/summary_by_dataset/<safe_dataset>.csv
    Row contains Model, Style, and the presentable score columns.
    """
    out_dir = os.path.join(target_dir, "summary_by_dataset")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{safe_dataset}.csv")

    row = {"Model": safe_model, "Style": style}
    row.update(presentable_scores)
    new_df = pd.DataFrame([row])

    if os.path.exists(path):
        old = pd.read_csv(path)
        # remove any prior row for this model before appending
        old = old[old["Model"] != safe_model]
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df

    # stable order: Model, Style, then scores sorted by name
    fixed_cols = ["Model", "Style"]
    score_cols = sorted([c for c in merged.columns if c not in fixed_cols])
    merged = merged[fixed_cols + score_cols]
    merged.to_csv(path, index=False, encoding="utf-8-sig")


def update_leaderboard(
    target_dir: str,
    safe_dataset: str,
    safe_model: str,
    overall_accuracy: float
) -> None:
    """
    Update a global table at <target>/leaderboard.csv
    Rows are models. Columns are datasets. Values are Average (percent).
    Adds a Mean column as rowwise mean across dataset columns.
    """
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, "leaderboard.csv")

    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=["Model"])

    # ensure a row for this model
    if safe_model not in df.get("Model", pd.Series(dtype=str)).values:
        df = pd.concat([df, pd.DataFrame([{"Model": safe_model}])], ignore_index=True)

    # set this dataset column
    df.loc[df["Model"] == safe_model, safe_dataset] = float(overall_accuracy)

    # recompute Mean across only dataset columns
    dataset_cols = [c for c in df.columns if c not in ["Model", "Mean"]]
    if dataset_cols:
        df["Mean"] = df[dataset_cols].mean(axis=1, numeric_only=True)

    # stable order: Model, dataset columns sorted, then Mean
    ordered = ["Model"] + sorted(dataset_cols)
    if "Mean" in df.columns:
        ordered += ["Mean"]
    df = df[ordered]

    df.to_csv(path, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a single dataset for a single model.")
    parser.add_argument("-s", dest="source", default="ko-r1-moreevals", help="input folder with CSV files")
    parser.add_argument("-t", dest="target", default="eval_result_1", help="output folder")
    parser.add_argument("-m", dest="model", required=True, help="model name of interest")
    parser.add_argument("-d", dest="dataset", required=True, help="dataset name of interest")
    parser.add_argument("-y", dest="style", required=True,
                        choices=["mcqa", "mcqa_raw", "open_math"],
                        help="evaluation style")
    args = parser.parse_args()

    infile = find_file(args.source, args.dataset, args.model)
    if infile is None:
        print(f"Error: could not find file for dataset='{args.dataset}' and model='{args.model}' in '{args.source}'")
        sys.exit(1)

    df = pd.read_csv(infile)

    if args.style == "mcqa":
        scores = evaluate_mcqa(df)
    elif args.style == "mcqa_raw":
        scores = evaluate_mcqa_raw(df)
    elif args.style == "open_math":
        scores = evaluate_open_math(df)
    else:
        print(f"Unknown style: {args.style}")
        sys.exit(2)

    safe_model = safe_name(args.model)
    safe_data = safe_name(args.dataset)
    out_csv = os.path.join(args.target, f"{safe_data}-{safe_model}.csv")
    write_scores_csv(scores, out_csv)
    
    # build presentable keys to reuse across summaries
    presentable = to_presentable_scores(scores)
    
    # per-dataset summary across models
    update_dataset_summary(
        target_dir=args.target,
        safe_dataset=safe_data,
        safe_model=safe_model,
        presentable_scores=presentable,
        style=args.style
    )
    
    # global leaderboard across datasets
    update_leaderboard(
        target_dir=args.target,
        safe_dataset=safe_data,
        safe_model=safe_model,
        overall_accuracy=scores.get("overall_accuracy", float("nan"))
    )


if __name__ == "__main__":
    main()
