#!/usr/bin/env python3
"""
reference audio (wav等) を Irodori-TTS の --ref-latent で使える .pt ファイルに変換するスクリプト。

使い方:
    uv run python scripts/encode_ref_latent.py --ref-wav path/to/reference.wav --output refs/my_voice.pt

内部で使っている irodori_tts.codec.DACVAECodec.load() / encode_file() は
irodori_tts/codec.py に実際に存在する公開インターフェースを使用しています。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# このスクリプトが scripts/ 配下など、リポジトリルート以外から実行されても
# irodori_tts パッケージを import できるように、リポジトリルートを sys.path に追加する。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from irodori_tts.codec import DACVAECodec


def detect_device() -> str:
    """cuda -> mps -> cpu の優先順位で利用可能なデバイスを自動判定する。"""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode a reference waveform into a .pt latent usable with infer.py --ref-latent."
    )
    parser.add_argument(
        "--ref-wav",
        required=True,
        help="変換したい reference audio ファイルのパス（wav, mp3等）",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="出力する .pt ファイルのパス（例: refs/my_voice.pt）",
    )
    parser.add_argument(
        "--codec-repo",
        default="Aratako/Semantic-DACVAE-Japanese-32dim",
        help="DACVAE コーデックの HuggingFace repo id。通常は変更不要。",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda / mps / cpu を明示的に指定したい場合。未指定なら自動判定。",
    )
    parser.add_argument(
        "--normalize-db",
        type=float,
        default=-16.0,
        help="ラウドネス正規化のターゲット値。infer.py の --ref-normalize-db デフォルトと揃えてあります。",
    )
    args = parser.parse_args()

    ref_wav_path = Path(args.ref_wav)
    if not ref_wav_path.exists():
        raise FileNotFoundError(f"reference audio が見つかりません: {ref_wav_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device or detect_device()
    print(f"[encode_ref_latent] device: {device}")
    print(f"[encode_ref_latent] codec_repo: {args.codec_repo}")

    codec = DACVAECodec.load(
        repo_id=args.codec_repo,
        device=device,
        normalize_db=args.normalize_db,
    )

    print(f"[encode_ref_latent] encoding: {ref_wav_path}")
    latent = codec.encode_file(str(ref_wav_path))
    print(f"[encode_ref_latent] latent shape: {tuple(latent.shape)}")

    torch.save(latent, output_path)
    print(f"[encode_ref_latent] saved: {output_path}")


if __name__ == "__main__":
    main()