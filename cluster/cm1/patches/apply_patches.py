#!/usr/bin/env python3
"""Apply our CM1 source patches to a copy of the source tree (never to cm1data/ itself).

Each patch is an anchored text replacement in one file: the anchor must occur exactly once, a patched
file is detected by the patch's marker comment and left alone (idempotent), and every patch leaves CM1
bit-for-bit unchanged when its controlling namelist variable (param8 spare var) is 0.

seed_var1 (init3d.F): a per-member seed for CM1's random initial noise.
  CM1 seeds the noise with a fixed constant (use_truly_random_pert = .false.), so every member gets the
  same noise. With this patch the namelist's spare variable var1 (param8) scrambles the seed when it is
  non-zero: var1 = 0 leaves CM1 bit-for-bit unchanged; each non-zero integer gives its own, reproducible
  noise.

two_storms_var2 (init3d.F, iinit = 8 only): two squall lines in one domain.
  iinit = 8 starts the storm with one warm line thermal (2 K, horizontal radius 10 km, vertical radius
  1.5 km, centred at z = 1.5 km and at x = centerx, the domain centre) plus +-0.2 K noise inside it.
  With var2 non-zero the one thermal is replaced by two identical ones centred at centerx -/+ var2/2,
  i.e. var2 is the separation (m) of the two storms, placed symmetrically about the domain centre.
  E.g. nx = 3200, dx = 250 m (800 km), var2 = 400000: thermals at x = 200 km and 600 km, so each storm
  sits 200 km from its outer boundary and owns a 400 km half-domain, exactly like the single storm in a
  400 km run. The noise needs no change: CM1 draws one random number per grid point from a single
  stream, so the second thermal (and the domain-wide irandp noise around it) already gets its own,
  independent draws. CM1 stops if the thermals overlap (|var2| < 2*bhrad) or leave the domain.
  var2 = 0 leaves CM1 bit-for-bit unchanged (the bubble code path and the random stream are untouched).

tiny_pert_var3 (init3d.F, any iinit): a tiny extra theta perturbation for twin runs (error growth).
  With var3 non-zero, a field uniform in [-var3, var3] K is added to the initial theta perturbation tha
  at every grid point of the domain (k = 1..nk, all heights, like irandp's +-0.25 K noise). It is added
  once, after all iinit options and after the irandp noise, i.e. on the same array (tha) and the same
  path as the irandp noise: the pressure balance below it and getset's boundary conditions / MPI halo
  exchange then see it exactly as they see that noise. Its values do NOT come from random_number (that
  would shift CM1's stream for everything drawn later) but from an integer hash (lowbias32, C. Wellons,
  chained over seed, i, j, k) of the GLOBAL grid indices, so the field is reproducible and the same for
  every MPI decomposition. var4 is its seed (integer; var4 = 0 -> a fixed default seed). Twin pair:
  control var3 = 0, perturbed var3 = 1e-3, everything else (var1, var4, ...) identical: both runs get
  exactly the same usual noise and differ only by the tiny field. var3 = 0 leaves CM1 bit-for-bit
  unchanged (the whole code path is skipped).

Usage: python3 apply_patches.py SRC_DIR
"""
import sys
from pathlib import Path

# --- seed_var1 ---------------------------------------------------------------------------------------
SEED_MARK = "! latent_DA patch seed_var1"
SEED_ANCHOR = "          sand(n) = nint( 2.0e9*(2.0*float(n-1)/float(k-1)-1.0) )\n        enddo\n"
SEED_PATCH = SEED_ANCHOR + f"""        {SEED_MARK}: non-zero var1 scrambles the seed (var1 = 0: unchanged CM1)
        if( nint(var1).ne.0 )then
          do n=1,k
            sand(n) = ieor( sand(n), int( mod( int(nint(var1),8)*2654435761_8 + int(n,8)*40503_8, &
                                               2147483647_8 ), 4 ) )
          enddo
          if(dowr) write(outfile,*) '  latent_DA: noise seed scrambled with var1 = ',nint(var1)
        endif
"""

# --- two_storms_var2 ---------------------------------------------------------------------------------
# Anchor: the iinit = 8 block from its noise amplitude through its bubble-distance formula (the formula
# alone also occurs in iinit = 1; with "amplitude = 0.20" in front it is unique).
STORM_MARK = "! latent_DA patch two_storms_var2"
STORM_HEAD = "        ! amplitude of random perturbations:\n        amplitude = 0.20\n"
STORM_LOOP = """
        do k=1,nk
        do jj=1,ny
        do ii=1,nx
          call random_number(rand)
          i = ii - myi1 + 1
          j = jj - myj1 + 1
        IF( i.ge.ib .and. i.le.ie .and. j.ge.jb .and. j.le.je )THEN
          beta=sqrt(                             &
                    ((xh(i)-ric)/bhrad)**2       &
                   +((zh(i,j,k)-zc)/bvrad)**2)
"""
STORM_ANCHOR = STORM_HEAD + STORM_LOOP
STORM_PATCH = STORM_HEAD + f"""
        {STORM_MARK}: non-zero var2 = separation (m) of two identical thermals
        ! at centerx -/+ var2/2 instead of one at centerx (var2 = 0: unchanged CM1)
        if( var2.ne.0.0 )then
          if( 0.5*abs(var2).lt.bhrad .or. ric-0.5*abs(var2)-bhrad.lt.minx .or.   &
              ric+0.5*abs(var2)+bhrad.gt.maxx )then
            if(myid.eq.0) print *,'  latent_DA two_storms_var2: thermals overlap or leave the domain:'
            if(myid.eq.0) print *,'  var2,bhrad,centerx,minx,maxx = ',var2,bhrad,ric,minx,maxx
            call stopcm1
          endif
          if(dowr) write(outfile,*) '  latent_DA: two thermals at x = ',ric-0.5*abs(var2),ric+0.5*abs(var2)
        endif
""" + STORM_LOOP + """          if( var2.ne.0.0 )then
            ! distance to the nearer of the two thermal centres
            beta=sqrt( min( ((xh(i)-(ric-0.5*abs(var2)))/bhrad)**2,            &
                            ((xh(i)-(ric+0.5*abs(var2)))/bhrad)**2 )           &
                      +((zh(i,j,k)-zc)/bvrad)**2 )
          endif
"""

# --- tiny_pert_var3 ----------------------------------------------------------------------------------
# Anchor: the start of the elliptic-solver section, right after the irandp block has closed (outside
# every iinit option and outside irandp, so it runs exactly once for any iinit / irandp / testcase).
# The patch goes in FRONT of the anchor. Exact mod-2**32 arithmetic in integer(8) (constants split into
# 16-bit halves, so no product exceeds 2**48): no reliance on signed overflow. Loop variables are local
# to a BLOCK so init3d's own i, j, k are untouched. Ghost points outside the physical domain are left to
# getset's bcs; halo points inside it get the owner's value (same global indices).
TINY_MARK = "! latent_DA patch tiny_pert_var3"
TINY_ANCHOR = "!----------------------------------------------\n!  arrays for elliptic solver\n"
TINY_PATCH = f"""      {TINY_MARK}: non-zero var3 adds a tiny theta field, uniform in
      ! [-var3,var3] K at every grid point, from an integer hash of the global grid indices and the
      ! seed var4 (0: fixed default), not from random_number, so the usual noise is unchanged
      ! (var3 = 0: unchanged CM1)
      IF( var3.ne.0.0 )THEN
        BLOCK
          integer(8) :: hsh,hkey(4)
          integer :: ip,jp,kp,ig,jg,m
          hkey(1) = 1013904223_8
          if( nint(var4).ne.0 ) hkey(1) = int(nint(var4),8)
          do kp=1,nk
          do jp=jb,je
          do ip=ib,ie
            ig = ip + myi1 - 1
            jg = jp + myj1 - 1
            IF( ig.ge.1 .and. ig.le.nx .and. jg.ge.1 .and. jg.le.ny )THEN
              hkey(2) = int(ig,8)
              hkey(3) = int(jg,8)
              hkey(4) = int(kp,8)
              hsh = 0_8
              do m=1,4
                ! lowbias32: x^=x>>16; x*=0x7feb352d; x^=x>>15; x*=0x846ca68b; x^=x>>16
                hsh = iand( ieor(hsh,hkey(m)), 4294967295_8 )
                hsh = ieor( hsh, ishft(hsh,-16) )
                hsh = iand( hsh*13613_8 + ishft(iand(hsh*32747_8,65535_8),16), 4294967295_8 )
                hsh = ieor( hsh, ishft(hsh,-15) )
                hsh = iand( hsh*42635_8 + ishft(iand(hsh*33900_8,65535_8),16), 4294967295_8 )
                hsh = ieor( hsh, ishft(hsh,-16) )
              enddo
              tha(ip,jp,kp) = tha(ip,jp,kp)    &
                  + var3*real( 2.0d0*(dble(hsh)+0.5d0)/4294967296.0d0 - 1.0d0 )
            ENDIF
          enddo
          enddo
          enddo
        END BLOCK
        if(dowr) write(outfile,*) '  latent_DA: tiny perturbation amplitude ', var3, ' seed ', var4
      ENDIF

""" + TINY_ANCHOR

PATCHES = [  # (name, file, marker, anchor, replacement)
    ("seed_var1", "init3d.F", SEED_MARK, SEED_ANCHOR, SEED_PATCH),
    ("two_storms_var2", "init3d.F", STORM_MARK, STORM_ANCHOR, STORM_PATCH),
    ("tiny_pert_var3", "init3d.F", TINY_MARK, TINY_ANCHOR, TINY_PATCH),
]


def apply(src: Path, name, fname, mark, anchor, patch):
    f = src / fname
    s = f.read_text()
    if mark in s:
        print(f"{f}: already patched ({name})")
        return
    if s.count(anchor) != 1:
        sys.exit(f"{f}: expected exactly one anchor for patch {name}, found {s.count(anchor)}")
    f.write_text(s.replace(anchor, patch))
    print(f"{f}: patched ({name})")


def main(src: Path):
    for p in PATCHES:
        apply(src, *p)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
