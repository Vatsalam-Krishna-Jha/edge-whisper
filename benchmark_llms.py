import csv
import gc
import os
import time
import psutil
from llama_cpp import Llama

MODELS = [
    {
        "name": "Llama-3.2-1B-Q4",
        "path": "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    },
    {
        "name": "Llama-3.2-3B-Q4",
        "path": "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    },
    {
        "name": "Phi-3-mini-Q4",
        "path": "models/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
    },
]

QUESTIONS = [
    ("factual",    "What is the capital of Australia?"),
    ("factual",    "Who painted the Mona Lisa?"),
    ("factual",    "What is the chemical symbol for gold?"),
    ("factual",    "How many planets are in our solar system?"),
    ("math",       "What is 47 multiplied by 13?"),
    ("math",       "If I have 3 apples and eat 2, how many do I have left?"),
    ("math",       "What is 15 percent of 200?"),
    ("define",     "Define photosynthesis in one sentence."),
    ("define",     "What is machine learning?"),
    ("define",     "What does HTTP stand for?"),
    ("commsense",  "Why do people wear sunglasses?"),
    ("commsense",  "What should I do if I smell smoke at home?"),
    ("commsense",  "Why is the sky blue?"),
    ("convo",      "Tell me a fun fact about octopuses."),
    ("convo",      "Suggest a healthy breakfast idea."),
]

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Always answer in under 40 words. Be direct and factually accurate."
)

OUTPUT_CSV = "results/benchmark.csv"


def get_rss_mb():
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 * 1024)


def benchmark_model(model_info):
    name = model_info["name"]
    path = model_info["path"]
    print(f"\n{'=' * 60}")
    print(f"Loading {name} ...")
    t0 = time.time()
    llm = Llama(model_path=path, n_ctx=2048, n_threads=4, verbose=False)
    load_time = time.time() - t0
    print(f"Loaded in {load_time:.1f}s. RAM: {get_rss_mb():.0f} MB")

    print("Warming up...")
    _ = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Say hi."},
        ],
        max_tokens=20,
        temperature=0.3,
    )
    print("Warm-up done. Starting benchmark.\n")

    rows = []
    for i, (category, q) in enumerate(QUESTIONS, start=1):
        print(f"  [{i:2d}/{len(QUESTIONS)}] {q}")
        t_start = time.time()
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            max_tokens=120,
            temperature=0.3,
        )
        total_time = time.time() - t_start
        answer = out["choices"][0]["message"]["content"].strip()
        usage = out.get("usage", {}) or {}
        completion_tokens = usage.get("completion_tokens", 0)
        prompt_tokens = usage.get("prompt_tokens", 0)
        tps = completion_tokens / total_time if total_time > 0 else 0
        ram = get_rss_mb()

        short = answer[:80] + ("..." if len(answer) > 80 else "")
        print(f"      A: {short}")
        print(f"      total_time={total_time:.2f}s  tokens={completion_tokens}  "
              f"tps={tps:.1f}  ram={ram:.0f}MB\n")

        rows.append({
            "model": name,
            "category": category,
            "question": q,
            "answer": answer,
            "total_time_s": round(total_time, 3),
            "completion_tokens": completion_tokens,
            "prompt_tokens": prompt_tokens,
            "tokens_per_second": round(tps, 2),
            "peak_ram_mb": round(ram, 1),
            "quality_score": "",
        })

    del llm
    gc.collect()
    return rows


def main():
    os.makedirs("results", exist_ok=True)
    all_rows = []
    for model_info in MODELS:
        rows = benchmark_model(model_info)
        all_rows.extend(rows)

    fieldnames = list(all_rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. Results written to {OUTPUT_CSV}")
    print(f"Total questions: {len(all_rows)}")


if __name__ == "__main__":
    main()
