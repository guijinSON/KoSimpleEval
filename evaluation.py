import argparse
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a vLLM model on a dataset and save its responses."
    )
    p.add_argument("--model", required=True,
                   help="Hugging Face model ID or local path")
    p.add_argument("--dataset", required=True,
                   help="Dataset name within HAERAE-HUB/KoSimpleEval")
    p.add_argument("--split", default="test",
                   help="Dataset split (default: test)")
    p.add_argument("--revision", default=None,
                   help="Model revision or commit hash (optional)")
    p.add_argument("--max_tokens", type=int, default=32768,
                   help="Maximum tokens to generate")
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Sampling temperature")
    p.add_argument("--top_p", type=float, default=1.0,
                   help="Top-p / nucleus sampling (1.0 disables)")
    p.add_argument("--output", default=None,
                   help="Output CSV path (auto-generated if omitted)")
    p.add_argument("--system_prompt", default="문제 풀이를 마친 후, 최종 정답을 다음 형식으로 작성해 주세요: \\boxed{N}.",
                   help="System prompt text to prepend to every conversation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- build LLM ----------------------------------------------------------
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": torch.cuda.device_count(),
    }
    if args.revision:
        llm_kwargs["revision"] = args.revision

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # ---- load data ----------------------------------------------------------
    df = load_dataset("HAERAE-HUB/KoSimpleEval", args.dataset, split=args.split).to_pandas()

    # ---- craft prompts ------------------------------------------------------
    prompts = []
    for _, row in df.iterrows():
        messages = []
        if args.system_prompt:
            messages.append({"role": "system", "content": args.system_prompt})
        messages.append({"role": "user", "content": row['question']})

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    # ---- generate -----------------------------------------------------------
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(prompts, sampling)
    df["response"] = [o.outputs[0].text for o in outputs]

    # ---- save ---------------------------------------------------------------
    if args.output is None:
        safe_model = args.model.replace("/", "_")
        safe_data = args.dataset.replace("/", "_")
        args.output = f"{safe_data}-{safe_model}.csv"

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows -> {args.output}")


if __name__ == "__main__":
    main()
