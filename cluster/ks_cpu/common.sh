#!/bin/bash
# Paper I/II models, CPU runs on Cashew (user 2026-11-14: the SwiGLU + RMSNorm mixer on the frozen Paper I/II autoencoders).
# Each run reuses a Paper I/II autoencoder (Stage A skipped: the checkpoint is copied to the run's tag), trains only the propagator.
. ~/venvs/cm1/bin/activate
cd "$HOME/latent_DA"
export PYTHONPATH=$PWD OMP_NUM_THREADS=${OMP_NUM_THREADS:-16} MKL_NUM_THREADS=${OMP_NUM_THREADS:-16}
A=artifacts
MIX="--site-mix-radius 0 --n-site-mix-layers 0 --dec-site-mix-radius 0 --dec-n-site-mix-layers 0 --local-channels 6"
# system -> AE checkpoint of the Paper I/II model, and the Stage B recipe of that model
kolm_ae=$A/stage1_ae_kolmogorov_crop_k_xs01_v2.pt
ks2d_ae=$A/stage1_ae_kolmogorov_crop_r13_C2_ks2d_s44big_k48_seed0.pt
ks1d_ae=$A/stage1_ae_patched_full_section224_ks_localfield_d48_localmlp.pt
kolm_args() { echo "--dataset $A/datasets/kolmogorov_wide16_alpha0.1.h5 $MIX --k-max 16 --k-mid 8 --noise-end 0.02 --prop-steps ${STEPS:-9000} --w-logdet 0.005 --crop-sites 48 --window 3 --prop-layers 2"; }
ks2d_args() { echo "--dataset $A/datasets/ks2d_box44_large.h5 --n-offsets 16 $MIX --k-max 48 --k-mid 24 --noise-end 0.02 --prop-steps ${STEPS:-12000} --crop-sites 8 --crop-sites-y 8 --window 3 --prop-layers 1"; }
# train_crop SYSTEM TAG VARIANT SEED: copy the AE under TAG (Stage A skipped), train Stage B on the CPU
train_crop() {
  local sys=$1 tag=$2 variant=$3 seed=$4 ae; eval ae=\$${sys}_ae
  cp "$ae" "$A/stage1_ae_kolmogorov_crop_$tag.pt"
  python scripts/train_kolmogorov_crop.py --tag "$tag" --seed "$seed" --device cpu --mixer-variant "$variant" $(${sys}_args)
}
