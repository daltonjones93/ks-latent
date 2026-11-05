"""D11: environment features of every member of an ensemble, from its input soundings (ks_latent/cm1/env.py).
  python scripts/cm1/build_env.py ~/cm1_bundle/ens_B ~/cm1_env/ens_B.json --prefix ens_B_
The prefix must match the member names in the cache (build_cache.py writes <ENS>_<member>; the first
ens_A cache used bare member names, so --prefix "" there).
"""
import argparse

from ks_latent.cm1.env import build_env_file

ap = argparse.ArgumentParser()
ap.add_argument("ens"); ap.add_argument("out"); ap.add_argument("--prefix", default="")
a = ap.parse_args()
d = build_env_file(a.ens, a.out, a.prefix)
print(f"{len(d)} members -> {a.out}")
