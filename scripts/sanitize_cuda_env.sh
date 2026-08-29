#!/usr/bin/env bash

# Keep training jobs isolated from platform CUDA paths injected by the container.
# The GR00T env uses PyTorch CUDA wheels, so those libraries must be searched
# before system CUDA/cuDNN paths.  The real driver path is appended later.

sanitize_cuda_env() {
  if [ -n "${CONDA_PREFIX:-}" ]; then
    unset LIBRARY_PATH
    export CUDA_HOME="$CONDA_PREFIX"
    export CUDA_PATH="$CONDA_PREFIX"
  fi

  local clean_ld=""
  local old_ifs="$IFS"
  local path_entry
  IFS=:
  for path_entry in ${LD_LIBRARY_PATH:-}; do
    case "$path_entry" in
      ""|*/compat|*/compat/|*/stubs|*/stubs/|/usr/local/cuda/lib64|/usr/local/cuda-*/lib64|/usr/lib/x86_64-linux-gnu|/usr/local/nvidia/lib|/usr/local/nvidia/lib64)
        continue
        ;;
    esac
    clean_ld="${clean_ld:+$clean_ld:}$path_entry"
  done
  IFS="$old_ifs"
  export LD_LIBRARY_PATH="$clean_ld"

  _prepend_ld_once() {
    [ -d "$1" ] || return 0
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$1:"*) ;;
      *) export LD_LIBRARY_PATH="$1${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  }

  if [ -n "${CONDA_PREFIX:-}" ]; then
    local site_packages="$CONDA_PREFIX/lib/python3.10/site-packages"
    _prepend_ld_once "$CONDA_PREFIX/targets/x86_64-linux/lib"
    _prepend_ld_once "$CONDA_PREFIX/lib"
    _prepend_ld_once "$site_packages/torch/lib"
    if [ -d "$site_packages/nvidia" ]; then
      local nvidia_lib
      for nvidia_lib in "$site_packages"/nvidia/*/lib; do
        _prepend_ld_once "$nvidia_lib"
      done
    fi
  fi

  _append_ld_once() {
    [ -d "$1" ] || return 0
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$1:"*) ;;
      *) export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$1" ;;
    esac
  }

  _append_ld_once /usr/lib/x86_64-linux-gnu
  _append_ld_once /usr/local/nvidia/lib
  _append_ld_once /usr/local/nvidia/lib64

  unset -f _prepend_ld_once _append_ld_once
}
