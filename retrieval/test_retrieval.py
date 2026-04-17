from retrieve_knowledge import retrieve_knowledge

# Example 1 — Choking
res = retrieve_knowledge(
    query="My child is choking",
    intent="choking",
    age_group="child"
)

print("\n--- CHOKING RESULTS ---")
for r in res:
    print(f"[{r['age_group']}] {r['text'][:200]}...\n")

# Example 2 — Bleeding
res = retrieve_knowledge(
    query="Someone is bleeding badly",
    intent="bleeding"
)

print("\n--- BLEEDING RESULTS ---")
for r in res:
    print(r["text"][:200], "\n")
