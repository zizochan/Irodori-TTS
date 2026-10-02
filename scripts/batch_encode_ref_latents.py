#!/usr/bin/env python3
"""
指定フォルダ内の mp3 / wav を一括で ref-latent (.pt) に変換するバッチスクリプト。
内部の変換処理は encode_ref_latent.py (同じ scripts/ 配下) の encode_reference() を使う。

使い方:
    uv run --no-sync python scripts/batch_encode_ref_latents.py <入力フォルダ>

    出力フォルダは省略可能。省略した場合は <入力フォルダ>/latents に自動作成される。

例:
    uv run --no-sync python scripts/batch_encode_ref_latents.py /path/to/ref_voices

    sample.mp3  -> /path/to/ref_voices/latents/sample.pt
    voice_a.wav -> /path/to/ref_voices/latents/voice_a.pt

    出力先を明示的に変えたい場合は --output-dir で指定できる:
    uv run --no-sync python scripts/batch_encode_ref_latents.py \\
        /path/to/ref_voices \\
        --output-dir /path/to/ref_latents

コーデックのロードは一度だけ行い、ファイルごとに使い回すので、
1ファイルずつ encode_ref_latent.py を呼び出すより高速です。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# このスクリプトが scripts/ 配下にある前提で、リポジトリルートを import パスに追加する。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# 同じ scripts/ 配下にある encode_ref_latent.py を import できるようにする。
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import torch

from encode_ref_latent import add_encode_options, encode_reference
from irodori_tts.codec import DACVAECodec
from irodori_tts.inference_runtime import default_runtime_device

AUDIO_EXTENSIONS = {".mp3", ".wav"}


def find_audio_files(input_dir: Path) -> list[Path]:
    files = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-encode all mp3/wav files in a folder into --ref-latent .pt files."
    )
    parser.add_argument(
        "input_dir",
        help="mp3 / wav が入っているフォルダ",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=".pt ファイルの出力先フォルダ（存在しなければ自動作成）。"
        "省略した場合は <入力フォルダ>/latents になる。",
    )
    add_encode_options(parser)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="出力先に同名の .pt が既にあっても上書きする（デフォルトはスキップ）。",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"入力フォルダが見つかりません: {input_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else (input_dir / "latents")

    audio_files = find_audio_files(input_dir)
    if not audio_files:
        print(f"[batch_encode] {input_dir} に mp3/wav が見つかりませんでした。")
        return

    print(f"[batch_encode] 対象ファイル数: {len(audio_files)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[batch_encode] 出力先: {output_dir}")

    device = args.device or default_runtime_device()
    print(f"[batch_encode] device: {device}")
    print(f"[batch_encode] codec_repo: {args.codec_repo}")

    # コーデックは一度だけロードして使い回す(ファイルごとの再ロードを避けて高速化)。
    codec = DACVAECodec.load(
        repo_id=args.codec_repo,
        device=device,
        normalize_db=args.normalize_db,
    )

    succeeded = 0
    skipped = 0
    failed: list[tuple[Path, str]] = []

    for i, audio_path in enumerate(audio_files, start=1):
        output_path = output_dir / f"{audio_path.stem}.pt"

        if output_path.exists() and not args.overwrite:
            print(f"[{i}/{len(audio_files)}] skip (既に存在): {audio_path.name}")
            skipped += 1
            continue

        print(f"[{i}/{len(audio_files)}] encoding: {audio_path.name}")
        start = time.time()
        try:
            latent = encode_reference(
                codec,
                audio_path,
                normalize_db=args.normalize_db,
                ensure_max=args.ensure_max,
                max_ref_seconds=args.max_ref_seconds,
                log=lambda msg: print(f"    {msg}"),
            )
            torch.save(latent, output_path)
        except Exception as e:  # noqa: BLE001 - バッチ処理なので1件失敗しても続行する
            print(f"    -> failed: {e}")
            failed.append((audio_path, str(e)))
            continue

        elapsed = time.time() - start
        print(
            f"    -> saved: {output_path.name} "
            f"(shape={tuple(latent.shape)}, {elapsed:.2f}s)"
        )
        succeeded += 1

    print()
    print("[batch_encode] 完了")
    print(f"  成功: {succeeded}")
    print(f"  スキップ（既存）: {skipped}")
    print(f"  失敗: {len(failed)}")
    if failed:
        for path, err in failed:
            print(f"    - {path.name}: {err}")


if __name__ == "__main__":
    main()
