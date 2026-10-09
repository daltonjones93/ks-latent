#!/bin/bash
while pgrep -f exp_crop_pipeline.py > /dev/null; do sleep 60; done
exec scripts/run_icnorm_ks2d_seeds.sh
