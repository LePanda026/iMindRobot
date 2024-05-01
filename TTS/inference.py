import sys, os
import logging
import torch
import utils
import numpy as np

import TTS.commons as commons
from TTS.models import SynthesizerTrn
from TTS.text.symbols import symbols
from TTS.text import cleaned_text_to_sequence, get_bert
from TTS.text.cleaner import clean_text

logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO, format="| %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


model = None

if sys.platform == "darwin" and torch.backends.mps.is_available():
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    device = "cuda"


def get_text(text, language_str, hps):
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str)
    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1
    bert = get_bert(norm_text, word2ph, language_str, device)
    del word2ph
    assert bert.shape[-1] == len(phone), phone
    if language_str == "ZH":
        bert = bert
        ja_bert = torch.zeros(768, len(phone))
    elif language_str == "JP":
        ja_bert = bert
        bert = torch.zeros(1024, len(phone))
    else:
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"
    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language


def infer(model, text, sdp_ratio, noise_scale, noise_scale_w, length_scale, sid, language,hps):
    bert, ja_bert, phones, tones, lang_ids = get_text(text, language, hps)
    with torch.no_grad():
        x_tst = phones.to(device).unsqueeze(0)
        tones = tones.to(device).unsqueeze(0)
        lang_ids = lang_ids.to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        ja_bert = ja_bert.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
        del phones
        speakers = torch.LongTensor([hps.data.spk2id[sid]]).to(device)
        audio = (
            model.infer(
                x_tst,
                x_tst_lengths,
                speakers,
                tones,
                lang_ids,
                bert,
                ja_bert,
                sdp_ratio=sdp_ratio,
                noise_scale=noise_scale,
                noise_scale_w=noise_scale_w,
                length_scale=length_scale,
            )[0][0, 0]
            .data.cpu()
            .float()
            .numpy()
        )
        del x_tst, tones, lang_ids, bert, x_tst_lengths, speakers
        torch.cuda.empty_cache()
        return audio


def tts_fn(model, text, speaker, sdp_ratio, noise_scale, noise_scale_w, length_scale, language,hps):
    slices = text.split("|")
    audio_list = []
    with torch.no_grad():
        for slice in slices:
            audio = infer(model,
                          slice,
                          sdp_ratio=sdp_ratio,
                          noise_scale=noise_scale,
                          noise_scale_w=noise_scale_w,
                          length_scale=length_scale,
                          sid=speaker,
                          language=language,
                          hps=hps)
            audio_list.append(audio)
            silence = np.zeros(hps.data.sampling_rate)  # 生成1秒的静音
            audio_list.append(silence)  # 将静音添加到列表中
    audio_concat = np.concatenate(audio_list)
    return "Success", (hps.data.sampling_rate, audio_concat)


def load_tts_hparams(hparams_path=r"D:\ChatGLM\Langchain-Chatchat\tts_config.json"):
    hps = utils.get_hparams_from_file(hparams_path)
    return hps


def load_tts_model(hps,checkpoint_path=r"D:\ChatGLM\Langchain-Chatchat\TTS\logs\YunJin\G_11000.pth"):
    device = ("cuda:0" if torch.cuda.is_available() else
              ("mps" if sys.platform == "darwin" and torch.backends.mps.is_available() else "cpu"))
    model = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to(device)
    _ = model.eval()
    _ = utils.load_checkpoint(checkpoint_path, model, None, skip_optimizer=True)
    return model


def get_audio_from_text(model, hps, text):
    _, audio_tuple = tts_fn(model,text, "Wanderer", 0.2, 0.6,
                            0.8, 1, "ZH",hps)
    return audio_tuple