import os
import logging

'''
%(asctime)s：这是一个时间占位符，它将被替换为记录日志消息的时间戳，通常是一个具有日期和时间的字符串。
%(filename)s：这是一个文件名占位符，它将被替换为记录日志消息的源文件的文件名。
line:%(lineno)d：这是一个行号占位符，它将被替换为记录日志消息的源文件中的行号。
%(levelname)s：这是一个日志级别占位符，它将被替换为记录日志消息的级别，如 "DEBUG"、"INFO"、"WARNING"、"ERROR" 或 "CRITICAL"。
%(message)s：这是一个消息占位符，它将被替换为实际的日志消息文本。
这个 LOG_FORMAT 字符串的目的是定义日志记录的外观和结构，使得日志消息可以包含有关时间、源文件、行号、日志级别和消息内容的信息。
当使用Python的标准日志模块（logging模块）来记录日志时，您可以将这个格式字符串传递给日志记录器，并根据需要将日志消息记录到文件、终端或其他地方，
以满足您的调试和记录需求。这有助于更容易地理解和分析日志，特别是在调试和故障排除过程中。
'''

# Set The prameters about log
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)
# 是否显示详细日志
log_verbose = False

tts_model_dict = {
    "bertvits2": {
        "japanese_ckpt": "./checkpoint/tts/bert/bert-base-japanese-v3",
        "chinese_ckpt": "./checkpoint/tts/bert/chinese-roberta-wwm-ext-large",
        "local_model_path": "./checkpoint/llm/chatglm2-6b",
        "api_base_url": "http://localhost:8888/v1",  # URL需要与运行fastchat服务端的server_config.FSCHAT_OPENAI_API一致
        "api_key": "EMPTY"
    },

}

TTS_MODEL = "bertvits2"