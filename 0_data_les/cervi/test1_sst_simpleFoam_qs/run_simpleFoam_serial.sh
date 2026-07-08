#!/bin/bash
set -e

OF=/homes/aiskhak/containers/of.sh

echo "Case: $(pwd)"
echo "Starting at: $(date)"

rm -rf postProcessing
rm -f log.run log.post.yPlus

$OF "simpleFoam > log.run 2>&1"

$OF "postProcess -func yPlus -latestTime > log.post.yPlus 2>&1" || \
$OF "simpleFoam -postProcess -func yPlus -latestTime > log.post.yPlus 2>&1"

echo "Done at: $(date)"
