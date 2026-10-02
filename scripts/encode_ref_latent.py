#!/usr/bin/env python3
"""
reference audio (wav等) を Irodori-TTS の --ref-latent / --ref-latents で使える
.pt ファイルに変換するスクリプト。

使い方:
    uv run --no-sync python scripts/encode_ref_latent.py \
        --ref-wav path/to/reference.wav --output refs/my_voice.pt

    # 複数クリップを使う場合は、1ファイルずつ変換して --ref-latents で渡す
    uv run --no-sync python infer.py ... --ref-latents refs/a.pt refs/b.pt

フォルダ内をまとめて変換する場合は batch_encode_ref_latents.py を使う。
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from pathlib import Path

# このスクリプトが scripts/ 配下など、リポジトリルート以外から実行されても
# irodori_tts パッケージを import できるように、リポジトリルートを sys.path に追加する。
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from irodori_tts.codec import DACVAECodec
from irodori_tts.inference_runtime import _load_audio, default_runtime_device


def parse_optional_float(value: str) -> float | None:
    """infer.py の --ref-normalize-db と同じく、none/off 等で None (無効) を受け付ける。"""
    raw = str(value).strip().lower()
    if raw in {"none", "null", "off", "disable", "disabled"}:
        return None
    try:
        out = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected float or one of [none, null, off, disable, disabled]."
        ) from exc
    if not math.isfinite(out):
        raise argparse.ArgumentTypeError(f"Expected finite float for value={value!r}.")
    return out


def add_encode_options(parser: argparse.ArgumentParser) -> None:
    """encode_ref_latent.py / batch_encode_ref_latents.py 共通のオプションを追加する。"""
    parser.add_argument(
        "--codec-repo",
        default="Aratako/Semantic-DACVAE-Japanese-32dim",
        help="DACVAE コーデックの HuggingFace repo id。通常は変更不要。",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda / mps / xpu / cpu を明示的に指定したい場合。未指定なら自動判定。",
    )
    parser.add_argument(
        "--normalize-db",
        type=parse_optional_float,
        default=-16.0,
        help=(
            "ラウドネス正規化のターゲット値。infer.py の --ref-normalize-db と同じ扱いで、"
            "'none' で無効化。デフォルト: -16.0"
        ),
    )
    parser.add_argument(
        "--ensure-max",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "正規化後にピークが 1.0 を超える場合のみスケールダウンする。"
            "infer.py の --ref-ensure-max と同じく、--normalize-db none のときのみ有効。"
            "デフォルト: 有効"
        ),
    )
    parser.add_argument(
        "--max-ref-seconds",
        type=float,
        default=None,
        help=(
            "指定した秒数で波形を先に切ってからエンコードする（infer.py の --ref-wav と同じ挙動）。"
            "チェックポイントの推奨値に合わせたいときに指定（旧モデルは30秒）。"
            "未指定または 0 以下なら切らない（推論時に latent 側で切られる）。"
        ),
    )


def encode_reference(
    codec: DACVAECodec,
    path: str | Path,
    *,
    normalize_db: float | None,
    ensure_max: bool = True,
    max_ref_seconds: float | None = None,
    log: Callable[[str], None] | None = None,
) -> torch.Tensor:
    """音声ファイルを (1, T_latent, D) の CPU テンソルにエンコードする。"""
    wav, sr = _load_audio(path)  # (C, T), sample_rate

    if max_ref_seconds is not None and max_ref_seconds > 0:
        max_samples = max(1, int(max_ref_seconds * float(sr)))
        if wav.shape[1] > max_samples:
            if log is not None:
                log(f"trimming {wav.shape[1] / sr:.2f}s -> {max_samples / sr:.2f}s")
            wav = wav[:, :max_samples]

    latent = codec.encode_waveform(
        wav.unsqueeze(0),  # (1, C, T)
        sample_rate=int(sr),
        normalize_db=normalize_db,
        ensure_max=bool(ensure_max),
    ).cpu()
    if latent.shape[1] == 0:
        raise ValueError(f"reference audio から空の latent が生成されました: {path}")
    return latent


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Encode a reference waveform into a .pt latent usable with "
            "infer.py --ref-latent / --ref-latents."
        )
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
    add_encode_options(parser)
    args = parser.parse_args()

    ref_wav_path = Path(args.ref_wav)
    if not ref_wav_path.exists():
        raise FileNotFoundError(f"reference audio が見つかりません: {ref_wav_path}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device or default_runtime_device()
    print(f"[encode_ref_latent] device: {device}")
    print(f"[encode_ref_latent] codec_repo: {args.codec_repo}")

    codec = DACVAECodec.load(
        repo_id=args.codec_repo,
        device=device,
        normalize_db=args.normalize_db,
    )

    print(f"[encode_ref_latent] encoding: {ref_wav_path}")
    latent = encode_reference(
        codec,
        ref_wav_path,
        normalize_db=args.normalize_db,
        ensure_max=args.ensure_max,
        max_ref_seconds=args.max_ref_seconds,
        log=lambda msg: print(f"[encode_ref_latent] {msg}"),
    )
    print(f"[encode_ref_latent] latent shape: {tuple(latent.shape)}")

    torch.save(latent, output_path)
    print(f"[encode_ref_latent] saved: {output_path}")


if __name__ == "__main__":
    main()
