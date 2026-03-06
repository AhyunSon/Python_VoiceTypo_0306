"""다국어 음소 인식 모델 기반 모음 분류.

facebook/wav2vec2-lv-60-espeak-cv-ft 사용.
IPA 음소를 직접 출력하므로, 한국어 모음에 해당하는
IPA 음소 확률을 읽어 모음 판별. 캘리브레이션 불필요.
"""

import json
import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
from huggingface_hub import hf_hub_download

# 한국어 모음 → IPA 매핑
VOWEL_IPA = {
    '아': 'a',
    '어': 'ʌ',
    '오': 'o',
    '우': 'u',
    '으': 'ɯ',
    '이': 'i',
    '에': 'e',
    '애': 'æ',
}


class PhonemeVowelDetector:
    def __init__(self, model_name: str = "facebook/wav2vec2-lv-60-espeak-cv-ft"):
        print(f"[phoneme] Loading model: {model_name}...", flush=True)
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self._model.eval()
        self._target_sr = 16000

        # vocab.json 직접 로드 (espeak 의존성 우회)
        vocab_path = hf_hub_download(model_name, "vocab.json")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        self._vowel_token_ids = {}  # {token_id: 한국어 모음}

        for kr_vowel, ipa in VOWEL_IPA.items():
            if ipa in vocab:
                self._vowel_token_ids[vocab[ipa]] = kr_vowel
                print(f"  {kr_vowel} → /{ipa}/ (token {vocab[ipa]})")
            else:
                print(f"  {kr_vowel} → /{ipa}/ NOT FOUND in vocab!")

        print(f"[phoneme] {len(self._vowel_token_ids)}/{len(VOWEL_IPA)} vowels mapped.")
        print("[phoneme] Model loaded.", flush=True)

    def get_vowel_probs(self, audio: np.ndarray, sr: int) -> dict:
        """오디오 → 모음별 확률.

        Returns:
            {모음: 확률} dict
        """
        # 리샘플링
        if sr != self._target_sr:
            ratio = self._target_sr / sr
            n_out = int(len(audio) * ratio)
            indices = np.arange(n_out) / ratio
            idx = np.clip(indices.astype(int), 0, len(audio) - 1)
            audio = audio[idx]

        inputs = self._feature_extractor(
            audio, sampling_rate=self._target_sr,
            return_tensors="pt", padding=False
        )

        with torch.no_grad():
            logits = self._model(**inputs).logits  # (1, T, vocab_size)

        probs = torch.softmax(logits, dim=-1).squeeze(0)  # (T, vocab_size)

        # 모음 토큰들의 프레임별 확률 합 → 모음 활성 프레임 찾기
        vowel_ids = list(self._vowel_token_ids.keys())
        vowel_frame_sum = probs[:, vowel_ids].sum(dim=1)  # (T,)

        # 모음 활성 상위 30% 프레임만 사용
        n_frames = len(vowel_frame_sum)
        k = max(1, n_frames // 3)
        top_indices = torch.topk(vowel_frame_sum, k).indices

        top_probs = probs[top_indices]  # (k, vocab_size)
        avg_top = top_probs.mean(dim=0)  # (vocab_size,)

        result = {}
        for tid, vowel in self._vowel_token_ids.items():
            result[vowel] = float(avg_top[tid])

        return result
