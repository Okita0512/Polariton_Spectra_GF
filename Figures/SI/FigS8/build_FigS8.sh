#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_rel="build"
build_dir="${script_dir}/${build_rel}"
source_rel="FigS8.tex"
output_rel="FigS8.pdf"

mkdir -p "${build_dir}"
cd "${script_dir}"

if [[ -n "${PDFLATEX:-}" ]]; then
  tex_engine="${PDFLATEX}"
elif command -v kpsewhich >/dev/null 2>&1 \
  && [[ -n "$(kpsewhich tikz.sty 2>/dev/null)" ]] \
  && [[ -n "$(kpsewhich standalone.cls 2>/dev/null)" ]]; then
  tex_engine="$(command -v pdflatex)"
elif command -v cmd.exe >/dev/null 2>&1 && command -v wslpath >/dev/null 2>&1; then
  windows_engine="$(cmd.exe /d /c 'where pdflatex' 2>/dev/null \
    | tr -d '\r' | sed -n '1p')"
  if [[ -z "${windows_engine}" ]]; then
    echo "Could not locate a pdflatex installation containing TikZ." >&2
    exit 1
  fi
  tex_engine="$(wslpath -u "${windows_engine}")"
else
  echo "Could not locate a pdflatex installation containing TikZ." >&2
  exit 1
fi

"${tex_engine}" \
  -interaction=nonstopmode \
  -halt-on-error \
  -file-line-error \
  -output-directory="${build_rel}" \
  "${source_rel}"

cp "${build_rel}/FigS8.pdf" "${output_rel}"
rm -f \
  "${build_rel}/FigS8.aux" \
  "${build_rel}/FigS8.log" \
  "${build_rel}/FigS8.pdf"
rmdir "${build_rel}" 2>/dev/null || true
echo "Wrote ${output_rel}"
