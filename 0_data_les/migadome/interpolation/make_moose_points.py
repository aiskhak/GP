import pandas as pd
import sys

inp = sys.argv[1]
out = sys.argv[2]

df = pd.read_csv(inp)
df = df.sort_values("id")

needed = ["id", "x", "y", "z", "elvol_aux_var", "yw_aux_var"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

df[needed].to_csv(
    out,
    sep=" ",
    header=False,
    index=False,
    float_format="%.16e",
)

print(f"Wrote {len(df)} points to {out}")
print("x range:", df.x.min(), df.x.max())
print("y range:", df.y.min(), df.y.max())
print("z range:", df.z.min(), df.z.max())
