#!/bin/bash
# Wait for 2018-2021 catfill to finish, then regenerate all plots.

cd /Users/mczhang/Documents/GitHub/Axial_Redpy
CONDA_BIN=/opt/miniconda3/envs/axial_autolocate_py311/bin

echo "$(date): Waiting for catfill (2018-2021) to finish..."
while pgrep -f "redpy-catfill" > /dev/null; do
    sleep 60
done
echo "$(date): Catfill finished. Regenerating all plots and HTML pages..."

$CONDA_BIN/redpy-force-plot -a -c axial_settings.cfg > plot_final.log 2>&1
echo "$(date): All done."
