"""Avoid rare multi-character tokens of the v4 Small text tokenizer at inference.

The modernbert-ja Unigram vocabulary keeps many long phrases as a single token
(e.g. "浦和レッズ", "ゼルダの伝説"). Such tokens are rare in the TTS training
data, so the model often misreads or skips them. At inference we encode with a
copy of the tokenizer in which those tokens are effectively disabled
(score -1e9); Unigram then falls back to shorter, well-trained pieces
("浦和|レッ|ズ"). The input text itself is never rewritten.

Rule (threshold measured on the modernbert-ja-310m score scale, so it is only
applied to models that use that tokenizer):
- disable tokens with >= 4 characters and log-prob score <= -13,
- except hiragana-only tokens (splitting them does not change the reading but
  breaks prosody) and KEEP_TOKENS,
- FORCE_SPLITS lists frequent tokens above the threshold that are still misread,
  with the split that was confirmed to be read correctly.

Listening tests on v4.1 Small: sentences with hard words went from 41% to 75%
read correctly; on ordinary sentences tokens +8%, audio length +3%, no new
misreadings.
"""

from __future__ import annotations

import json
import re

SPLIT_TOKENIZER_REPO = "sbintuitions/modernbert-ja-310m"
SCORE_THRESHOLD = -13.0
MIN_SPLIT_CHARS = 4

# Below the threshold, but misread once split (or the split breaks prosody).
KEEP_TOKENS = frozenset(
    {
        "心筋梗塞",
        "日常茶飯事",
        "瀬戸内海",
        "老若男女",
        "興味津々",
        "高田馬場",
        "アクリル板",
        "年モデル",
        "オフショア",
        "モスクワ",
        "テラス席",
        "缶バッジ",
        "ジュリー",
        "ッチャー",
        "ティーン",
        "トゥルー",
        "マリッジ",
        "リフィル",
        "リベット",
        "デパート",
        "カリウム",
        "タピオカ",
        "ボーイズ",
        "フレックス",
        "フリース",
        "ファブリック",
        "ユナイテッド",
        "ブルース",
        "ミスター",
        "タイムズ",
        "ソケット",
        "コスチューム",
        "プレビュー",
        "リーズナブル",
        "プロダクション",
        "ファイト",
        "アバター",
        "産婦人科",
        "行方不明",
        "直射日光",
    }
)

# Above the threshold, but misread as one token; split exactly like this.
FORCE_SPLITS = {
    "セブンイレブン": ("セブン", "イレブン"),
    "楽天市場": ("楽天", "市場"),
    "外壁塗装": ("外壁", "塗装"),
    "イベルメクチン": ("イベルメ", "クチン"),
    "ミノキシジル": ("ミノ", "キシ", "ジル"),
    "送料無料": ("送料", "無料"),
    "ヴィンテージ": ("ヴィン", "テージ"),
    "点セット": ("点", "セット"),
    "フォルダ": ("フォル", "ダ"),
    "消費者金融": ("消費者", "金融"),
    "代表取締役": ("代表", "取締役"),
    "本セット": ("本", "セット"),
    "厚生労働省": ("厚生", "労働", "省"),
    "オンラインカジノ": ("オンライン", "カジノ"),
    "当サイト": ("当", "サイト"),
    "プロペシア": ("プロ", "ペ", "シア"),
    "リノベーション": ("リノベ", "ー", "ション"),
    "メール便": ("メール", "便"),
    "不動産投資": ("不動産", "投資"),
    "ティック": ("ティ", "ック"),
    "フジテレビ": ("フジ", "テレビ"),
    "東日本大震災": ("東日本", "大震災"),
    "新品未使用": ("新品", "未使用"),
    "ネクスガード": ("ネ", "クス", "ガード"),
    "全身脱毛": ("全身", "脱毛"),
    "債務整理": ("債務", "整理"),
    "ス感染症": ("ス", "感染症"),
    "代表取締役社長": ("代表", "取締役", "社長"),
    "枚セット": ("枚", "セット"),
    "フィナステリド": ("フィ", "ナス", "テリ", "ド"),
    "クチコミ": ("クチ", "コミ"),
    "ブロガー": ("ブロ", "ガー"),
    "ルイヴィトン": ("ルイ", "ヴィ", "トン"),
    "転職エージェント": ("転職", "エージェント"),
    "数量限定": ("数量", "限定"),
    "ホワイトニング": ("ホワイト", "ニング"),
    "無料ダウンロード": ("無料", "ダウンロード"),
    "スーパーコピー": ("スーパー", "コピー"),
    "一般社団法人": ("一般", "社", "団", "法人"),
    "日本語版": ("日本語", "版"),
    "ペット用品": ("ペット", "用品"),
    "投資信託": ("投資", "信託"),
    "脱毛サロン": ("脱毛", "サロン"),
    "東京都港区": ("東京都", "港区"),
    "注文住宅": ("注文", "住宅"),
    "マイナビ": ("マイ", "ナビ"),
    "行政書士": ("行政", "書", "士"),
    "無料相談": ("無料", "相談"),
    "市区町村": ("市", "区", "町", "村"),
    "ザガーロ": ("ザ", "ガー", "ロ"),
    "リハビリテーショ": ("リハビリ", "テーショ"),
    "株主優待": ("株主", "優待"),
    "オンラインストア": ("オンライン", "ストア"),
    "プレゼンテーショ": ("プレゼン", "テーショ"),
    "個人事業主": ("個人", "事業主"),
    "社会福祉法人": ("社会", "福祉", "法人"),
    "カルティエ": ("カル", "ティエ"),
    "イオンモール": ("イオン", "モール"),
    "ピーター": ("ピー", "ター"),
    "銀行振込": ("銀行", "振込"),
    "スタッフ一同": ("スタッフ", "一同"),
    "即日融資": ("即日", "融資"),
    "プラセンタ": ("プラ", "センタ"),
    "大学大学院": ("大学", "大学院"),
    "モンクレール": ("モン", "ク", "レール"),
    "賃貸物件": ("賃貸", "物件"),
    "協同組合": ("協同", "組合"),
    "年以上前": ("年", "以上", "前"),
    "土日祝日": ("土日", "祝日"),
    "ウェイト": ("ウェ", "イト"),
}

_HIRAGANA_ONLY = re.compile(r"[ぁ-ゖゝゞー]+")


def _disabled(piece: str, score: float) -> bool:
    text = piece.replace("▁", "")
    return (
        score <= SCORE_THRESHOLD
        and len(text) >= MIN_SPLIT_CHARS
        and text not in KEEP_TOKENS
        and text not in FORCE_SPLITS
        and not _HIRAGANA_ONLY.fullmatch(text)
    )


class RareTokenSplitter:
    """Encode text without the rare long tokens of a Unigram `tokenizers.Tokenizer`."""

    def __init__(self, backend_tokenizer) -> None:
        from tokenizers import Tokenizer

        data = json.loads(backend_tokenizer.to_str())
        model = data.get("model", {})
        if model.get("type") != "Unigram" or not model.get("vocab"):
            raise ValueError("Rare token splitting requires a Unigram tokenizer.")
        for item in model["vocab"]:
            if _disabled(item[0], float(item[1])):
                item[1] = -1e9
        self.tokenizer = Tokenizer.from_str(json.dumps(data, ensure_ascii=False))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        # token id -> ids of its pieces, for FORCE_SPLITS words present in the vocabulary.
        self.forced: dict[int, list[int]] = {}
        for word, pieces in FORCE_SPLITS.items():
            word_id = self.tokenizer.token_to_id(word)
            if word_id is None:
                continue
            ids, length = [], 0
            for piece in pieces:
                piece_id = self.tokenizer.token_to_id(piece)
                if piece_id is not None:
                    ids.append(piece_id)
                    length += len(piece)
                    continue
                encoding = self.tokenizer.encode(piece, add_special_tokens=False)
                for token_id, (start, end) in zip(encoding.ids, encoding.offsets, strict=True):
                    if start != end:  # skip a bare "▁" the pre-tokenizer may add
                        ids.append(token_id)
                        length += end - start
            if length == len(word):
                self.forced[word_id] = ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for token_id in self.tokenizer.encode(text, add_special_tokens=False).ids:
            ids.extend(self.forced.get(token_id, (token_id,)))
        return ids
