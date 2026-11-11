#!/usr/bin/env python3
# =============================================================================
#   ai4earthquake :: scripts
# =============================================================================
#   Module       :: 02_embed.py -- Stage 2 (Module 2): delay embedding of each statistic, PCA(3) + UMAP(2)
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""Stage 2 (Module 2): delay embedding of each statistic, PCA(3) + UMAP(2).

Reads outputs/<system>/01_features/features.h5 and writes one file per
channel to outputs/<system>/02_embedding/:
    var.h5, kurt.h5   keys "train"/"test": time, value, target, PCA1-3, Emb1-2
and, with --target, the target-side reconstruction of Fig. 3:
    target.h5         the same columns for the target series itself

The split (last 20 % of the rows in the laboratory; by date at Kīlauea),
the stride S and the embedding dimension D come from the configuration.
"""
from common import parse
from labquake import embedding, io


def extra(p):
    p.add_argument("--target", action="store_true",
                   help="also embed the target series (target-side reconstruction)")
    p.add_argument("--stats", nargs="+", default=["var", "kurt"])


args, cfg = parse(__doc__, extra)
rec, feats = io.load_features(io.stage_dir(cfg, "01_features") / "features.h5")
out = io.stage_dir(cfg, "02_embedding")
emb = cfg["embedding"]
print(f"{cfg['system']}: S = {emb['S']:,}, D = {emb['D']}, split = {cfg['split']}")

for stat in args.stats:
    train, test, models = embedding.embed_channel(rec, feats, stat, cfg)
    io.save_table(train, out / f"{stat}.h5", "train", mode="w")
    io.save_table(test, out / f"{stat}.h5", "test", mode="a")
    evr = models["pca"].explained_variance_ratio_
    print(f"  {stat}: train {len(train):,} rows, test {len(test):,} rows, "
          f"PCA explained variance {evr.round(3)}")

if args.target:
    train, test, models = embedding.embed_target(rec, feats, cfg)
    io.save_table(train, out / "target.h5", "train", mode="w")
    io.save_table(test, out / "target.h5", "test", mode="a")
    print(f"  target: train {len(train):,} rows, test {len(test):,} rows")
