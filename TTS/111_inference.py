from inference import load_tts_hparams
from inference import load_tts_model
from inference import get_audio_from_text

hps = load_tts_hparams()
model = load_tts_model(hps)
audio = get_audio_from_text(model,hps,text="我是神里绫华的狗！")
