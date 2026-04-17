from datasets import load_dataset

# 📥 Load only the Superdataset split
ds = load_dataset("lextale/FirstAidInstructionsDataset", split="Superdataset")

print(ds)
print("🔑 Columns:", ds.column_names)
print("📝 First row:", ds[0])

# 💾 Save locally
ds.to_json("data/english/superdataset.json", orient="records", lines=True)
print("✅ Superdataset saved to data/english/superdataset.json")
