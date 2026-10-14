#!/usr/bin/env bash
# Steps of the experiment setup that do not need root. Run from the repo root:
#
#   bash exp/setup/user_setup.sh
#
# Installs, all under $HOME:
#   - Miniforge and a conda env with Python 3.12, nvcc 11.8 and gcc 11
#     (gsplat compiles its CUDA kernels on first use; nvcc 11.8 rejects
#     the system gcc 13, and the system Python has no headers);
#   - the packages in exp/requirements.txt and Playwright's Chromium;
#   - a static ffmpeg build with libvmaf;
#   - the pretrained 3DGS models, linked into static/models/<scene>/.
#
# GitHub downloads are very slow on some machines. Put the Miniforge
# installer or the ffmpeg tarball in $PKG_DIR first to skip downloading them.
set -euo pipefail

CONDA_DIR=${CONDA_DIR:-$HOME/miniforge3}
CONDA_ENV=${CONDA_ENV:-tigas}
OPT_DIR=${OPT_DIR:-$HOME/opt}
DATA_DIR=${DATA_DIR:-$HOME/tigas-data}
PKG_DIR=${PKG_DIR:-$DATA_DIR/pkgs}
SCENES=${SCENES:-"train truck room bicycle stump treehill playroom drjohnson"}

MINIFORGE=Miniforge3-Linux-x86_64.sh
FFMPEG_TAR=ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz
MODELS_URL=https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip

mkdir -p "$PKG_DIR" "$OPT_DIR" "$DATA_DIR"

fetch() {  # url, file in $PKG_DIR
    [[ -s $PKG_DIR/$2 ]] || curl -fL -C - -o "$PKG_DIR/$2" "$1"
}

echo "== Miniforge"
if [[ ! -x $CONDA_DIR/bin/conda ]]; then
    fetch https://github.com/conda-forge/miniforge/releases/latest/download/$MINIFORGE $MINIFORGE
    bash "$PKG_DIR/$MINIFORGE" -b -p "$CONDA_DIR"
fi
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"

echo "== conda env $CONDA_ENV"
if ! conda env list | grep -qE "^$CONDA_ENV "; then
    conda create -y -q -n "$CONDA_ENV" --override-channels \
        -c nvidia/label/cuda-11.8.0 -c conda-forge \
        python=3.12 cuda-nvcc=11.8 cuda-cudart-dev=11.8 cuda-cccl=11.8 \
        gcc_linux-64=11 gxx_linux-64=11
fi
# The compiler activation scripts read unset variables; set -u would abort.
set +u
conda activate "$CONDA_ENV"
set -u

echo "== pip packages"
pip install -q -r exp/requirements.txt
python -m playwright install chromium

echo "== ffmpeg"
if [[ ! -x $OPT_DIR/ffmpeg/bin/ffmpeg ]]; then
    fetch https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/$FFMPEG_TAR $FFMPEG_TAR
    tar xJf "$PKG_DIR/$FFMPEG_TAR" -C "$OPT_DIR"
    ln -sfn "$OPT_DIR/${FFMPEG_TAR%.tar.xz}" "$OPT_DIR/ffmpeg"
fi
"$OPT_DIR/ffmpeg/bin/ffmpeg" -version | head -1

echo "== 3DGS models"
if [[ ! -d $DATA_DIR/3dgs-models/train ]]; then
    fetch "$MODELS_URL" models.zip   # 14.7 GB
    python -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
        "$PKG_DIR/models.zip" "$DATA_DIR/3dgs-models"
fi
for s in $SCENES; do
    mkdir -p "static/models/$s"
    ln -sfn "$DATA_DIR/3dgs-models/$s/point_cloud/iteration_30000/point_cloud.ply" \
        "static/models/$s/point_cloud.ply"
done
ls static/models

echo "done; next: sudo bash exp/setup/root_setup.sh <rattan binary>, then bash exp/setup/check_env.sh"
