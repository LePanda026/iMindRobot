# iMindRobot
## 🔧 Quick Start
* Prepare virtual environment
```bash
conda create -n chatbot python=3.8
conda activate chatbot
```

* Clone our repository
```bash
git clone https://github.com/LePanda026/ChatBot.git
cd UniQ4Cap
```

* Install required packages:
```bash
cd requirements
pip install -r requirements_tts.txt
pip install -r requirements.txt
pip install -r requirements_api.txt
pip install -r requirements_webui.txt
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install streamlit_webrtc==1.26.0
pip install streamlit==1.26.0
pip install --upgrade langchain
```

* Download the Checkpoint and modify the Model Path :
```bash
cd requirements  
pip install -r requirements.txt
pip install -r requirements_api.txt
pip install -r requirements_webui.txt
pip install -r requirements_tts.txt

```

* Download the model:
All checkpoints can be availible here, and you should arrange them as following picture:
<img src="https://github.com/LePanda026/iMindRobot/blob/main/checkpoint.png" />  
ChatGLM2-6B: 🤗[Huggfacing](https://huggingface.co/THUDM/chatglm2-6b);  
TTS_MODEL: [Baidu Disk](https://pan.baidu.com/s/1-JsqKEBr2nl7VkhWFcOQgQ?pwd=void);  
Bert_Japanese: 🤗[Huggfacing](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3);  
Bert_Chinese: 🤗[Huggfacing](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large);  
* QuickStart
```bash
cd ..
python init_database.py
python startup.py -a
```


## 💖 Acknowledgement
* [ChatGLM](https://github.com/thudm/chatglm2-6b) ChatGLM2-6B: An Open Bilingual Chat.
* [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) Langchain-Chatchat (formerly langchain-ChatGLM), local knowledge based LLM (like ChatGLM) QA app with langchain.
* [Bert-Vits2](https://github.com/fishaudio/Bert-VITS2) vits2 backbone with multilingual-bert.
