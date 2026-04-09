#!/bin/bash
# Usage: bash scripts/use_preset.sh <preset_name>
# Copies preset params into params/ for the next run.
# Must be run from src/pulsesuite/PSTD3D/

PRESET="$1"
DIR="params/presets/$PRESET"

if [ -z "$PRESET" ]; then
    echo "Available presets:"
    echo ""
    echo "  === Propagation (standard 800nm / 5fs / 1.25e8 V/m in AlAs) ==="
    echo ""
    echo "    1D (fast):"
    echo "      propagation_1d   4096x1x1,    100 fs"
    echo ""
    echo "    2D (x-y plane, Gaussian beam visible in transverse):"
    echo "      propagation_2d   2048x256x1,   50 fs"
    echo ""
    echo "    3D (full volume):"
    echo "      propagation_3d   1024x128x128,  25 fs"
    echo ""
    echo "  === Physics variations -- 1D ==="
    echo ""
    echo "      broadband        Tw=2 fs     sub-2-cycle, ultra-wide spectrum"
    echo "      narrowband       Tw=20 fs    ~7.5-cycle, narrow spectrum"
    echo "      chirped          C=5e28      positive chirp, red-to-blue sweep"
    echo "      strongfield      Amp=1.25 GV/m   10x field strength"
    echo ""
    echo "  === Physics variations -- 2D ==="
    echo ""
    echo "      broadband_2d     chirped_2d     narrowband_2d     strongfield_2d"
    echo ""
    echo "  === Quick-test (1D) ==="
    echo ""
    echo "      smoke            128x1x1,   500 as"
    echo "      quick            512x1x1,    75 fs"
    echo "      medium          2048x1x1,   100 fs"
    echo "      production      4096x1x1,   140 fs"
    echo ""
    echo "  === Diagnostic ==="
    echo ""
    echo "      nopml            512x1x1,    21 fs   periodic BC (energy test)"
    echo ""
    echo "Usage: bash scripts/use_preset.sh <preset_name>"
    exit 0
fi

if [ ! -d "$DIR" ]; then
    echo "Error: preset '$PRESET' not found in $DIR"
    echo "Run without arguments to see available presets."
    exit 1
fi

cp "$DIR/space.params" params/space.params
cp "$DIR/time.params"  params/time.params
cp "$DIR/pulse.params" params/pulse.params

echo "Loaded preset: $PRESET"
echo "  space.params -> $(head -2 params/space.params | tail -1 | awk '{print $1}') x $(head -3 params/space.params | tail -1 | awk '{print $1}') x $(head -4 params/space.params | tail -1 | awk '{print $1}')"
echo "  time.params  -> t0=$(head -1 params/time.params | awk '{print $1}'), tf=$(head -2 params/time.params | tail -1 | awk '{print $1}')"
echo "  pulse.params -> Tw=$(head -3 params/pulse.params | tail -1 | awk '{print $1}'), Amp=$(head -2 params/pulse.params | tail -1 | awk '{print $1}')"
