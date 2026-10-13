#!/bin/bash
# Resume the jobs paused on 2026-10-12 21:17 to cut heat/load (Block Z).
kill -CONT 13849 23255 24443 26715 26717 26718 26719 26720 26721 27214 27376 27470 27519 27545 
kill -CONT 28144 28154 22397  # 21:24 pause: two R7 lanes, p7b 440^2 filter
