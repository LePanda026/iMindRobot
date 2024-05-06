# iMindRobot
## 🔧 Quick Start
* Prepare virtual environment
```bash
conda create -n imindrobot python=3.8
conda activate imindrobot
```

* Clone our repository
```bash
git clone https://github.com/LePanda026/iMindRobot.git
cd iMindRobot
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
pip install streamlit_webrtc
pip install streamlit==1.26.0
pip install --upgrade langchain
```

* Download the model:
All checkpoints can be availible here, and you should arrange them as following picture:
M3E: 🤗[Huggfacing](https://huggingface.co/moka-ai/m3e-base);  
ChatGLM2-6B: 🤗[Huggfacing](https://huggingface.co/THUDM/chatglm2-6b);  
Bert_Japanese: 🤗[Huggfacing](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3);  
Bert_Chinese: 🤗[Huggfacing](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large);  
TTS_MODEL(character): [Baidu Disk](https://pan.baidu.com/s/1-JsqKEBr2nl7VkhWFcOQgQ?pwd=void);  


<img src="https://github.com/LePanda026/iMindRobot/blob/main/checkpoint_arrangement.png" />  

* Start serve and webui
```bash
cd ..
```
Run following command to init database
```bash
python init_database.py
```
Run following command to start webui and api serve
```bash
python startup.py -a
```

## API
How to use api? You can reference to `http://127.0.0.1:7861` after starting serve and webui.  
For using api in Python, you can reference files under iMindRobot/tests/  
<img src="https://github.com/LePanda026/iMindRobot/blob/main/api_test.png" />  


## 💖 Acknowledgement
* [ChatGLM](https://github.com/thudm/chatglm2-6b) ChatGLM2-6B: An Open Bilingual Chat.
* [Langchain-Chatchat](https://github.com/chatchat-space/Langchain-Chatchat) Langchain-Chatchat (formerly langchain-ChatGLM), local knowledge based LLM (like ChatGLM) QA app with langchain.
* [Bert-Vits2](https://github.com/fishaudio/Bert-VITS2) vits2 backbone with multilingual-bert.
