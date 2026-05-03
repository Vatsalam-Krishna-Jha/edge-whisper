import csv
import os
import sys

CSV_PATH = "results/benchmark.csv"


def main():
    print("\n" + "=" * 70)
    print("QUALITY SCORING")
    print("=" * 70)
    print("""
Rate each answer on a scale of 1 to 5:
  1 = wrong, useless, or hallucinated
  2 = partially correct or confused
  3 = mostly correct, somewhat coherent
  4 = correct and clear
  5 = correct, clear, and well-written

Keys:
  1-5 = assign score
  s   = skip this answer
  q   = quit and save progress
""")
    input("Press Enter to start...")

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys())

    for i, row in enumerate(rows, start=1):
        if row.get("quality_score", "").strip():
            continue

        os.system("clear" if os.name == "posix" else "cls")
        print(f"\n[{i}/{len(rows)}]  Model: {row['model']}    "
              f"Category: {row['category']}")
        print("-" * 70)
        print(f"Q: {row['question']}")
        print()
        print(f"A: {row['answer']}")
        print("-" * 70)
        print(f"  time={row['total_time_s']}s  "
              f"tokens={row['completion_tokens']}  "
              f"tps={row['tokens_per_second']}")
        print()

        while True:
            choice = input("Score (1-5, s=skip, q=quit): ").strip().lower()
            if choice in ("1", "2", "3", "4", "5"):
                row["quality_score"] = choice
                break
            elif choice == "s":
                break
            elif choice == "q":
                print("Saving and quitting...")
                with open(CSV_PATH, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                sys.exit(0)
            else:
                print("Invalid input. Enter 1-5, s, or q.")

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\nAll scored! Saved to", CSV_PATH)


if __name__ == "__main__":
    main()
