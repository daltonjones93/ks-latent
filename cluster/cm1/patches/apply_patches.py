#!/usr/bin/env python3
"""Apply our CM1 source patches to a copy of the source tree (never to cm1data/ itself).

seed_var1 (init3d.F): a per-member seed for CM1's random initial noise.
  CM1 seeds the noise with a fixed constant (use_truly_random_pert = .false.), so every member gets the
  same noise. With this patch the namelist's spare variable var1 (param8) scrambles the seed when it is
  non-zero: var1 = 0 leaves CM1 bit-for-bit unchanged; each non-zero integer gives its own, reproducible
  noise. Idempotent: a patched file is detected by its marker and left alone.
Usage: python3 apply_patches.py SRC_DIR
"""
import sys
from pathlib import Path

MARK = "! latent_DA patch seed_var1"
ANCHOR = "          sand(n) = nint( 2.0e9*(2.0*float(n-1)/float(k-1)-1.0) )\n        enddo\n"
PATCH = ANCHOR + f"""        {MARK}: non-zero var1 scrambles the seed (var1 = 0: unchanged CM1)
        if( nint(var1).ne.0 )then
          do n=1,k
            sand(n) = ieor( sand(n), int( mod( int(nint(var1),8)*2654435761_8 + int(n,8)*40503_8, &
                                               2147483647_8 ), 4 ) )
          enddo
          if(dowr) write(outfile,*) '  latent_DA: noise seed scrambled with var1 = ',nint(var1)
        endif
"""


def main(src: Path):
    f = src / "init3d.F"
    s = f.read_text()
    if MARK in s:
        print(f"{f}: already patched")
        return
    if s.count(ANCHOR) != 1:
        sys.exit(f"{f}: expected exactly one seeding loop to patch, found {s.count(ANCHOR)}")
    f.write_text(s.replace(ANCHOR, PATCH))
    print(f"{f}: patched (seed_var1)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
