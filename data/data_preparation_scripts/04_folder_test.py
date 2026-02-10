from pathlib import Path

root = Path(r".\data\02__italy_fwg_outputs").resolve()
epws = sorted(root.rglob("*.epw"))

print("Root:", root)
print("Count:", len(epws))
print("First 10:")
for p in epws[:10]:
    print(" -", p)
