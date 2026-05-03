import time
from llama_cpp import Llama

MODEL_PATH = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

QUESTIONS = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "What is 17 times 23?",
]

SYSTEM_PROMPT = (
    "You are a helpful assistant. Keep answers under 30 words."
)


def main():
    print(f"Loading model from {MODEL_PATH} ...")
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4, verbose=False)
    print("Loaded.\n")

    for q in QUESTIONS:
        print(f"Q: {q}")
        t0 = time.time()
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            max_tokens=80,
            temperature=0.3,
        )
        elapsed = time.time() - t0
        answer = out["choices"][0]["message"]["content"].strip()
        print(f"A: {answer}")
        print(f"   ({elapsed:.1f}s)\n")


if __name__ == "__main__":
    main()
