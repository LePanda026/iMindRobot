import random
import time
import queue
import pydub
from streamlit_webrtc import WebRtcMode, webrtc_streamer
import streamlit as st
from configs.server_config import FSCHAT_MODEL_WORKERS
from webui_pages.utils import *
from streamlit_chatbox import *
from datetime import datetime
from server.chat.search_engine_chat import SEARCH_ENGINES
import os
from configs.model_config import LLM_MODEL, TEMPERATURE
from server.utils import get_model_worker_config
from typing import List, Dict

from TTS.inference import load_tts_hparams
from TTS.inference import load_tts_model
from TTS.inference import get_audio_from_text

from configs.model_config import tts_model_dict,TTS_MODEL
config_file = tts_model_dict[TTS_MODEL]['config_file']
model_file = tts_model_dict[TTS_MODEL]['model_file']
hps = load_tts_hparams(config_file)
model = load_tts_model(hps, model_file)

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "chatchat_icon_blue_square_v2.png"
    )
)

def get_messages_history(history_len: int) -> List[Dict]:
    def filter(msg):
        content = [x._content for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        return {
            "role": msg["role"],
            "content": content[0] if content else "",
        }

    history = chat_box.filter_history(100000, filter)  # workaround before upgrading streamlit-chatbox.
    user_count = 0
    i = 1
    for i in range(1, len(history) + 1):
        if history[-i]["role"] == "user":
            user_count += 1
            if user_count >= history_len:
                break
    return history[-i:]

def dialogue_page(api: ApiRequest):
    chat_box.init_session()

    with st.sidebar:
        def on_mode_change():
            mode = st.session_state.dialogue_mode
            text = f"已切换到 {mode} 模式。"
            if mode == "知识库问答":
                cur_kb = st.session_state.get("selected_kb")
                if cur_kb:
                    text = f"{text} 当前知识库： `{cur_kb}`。"
            st.toast(text)
            # sac.alert(text, description="descp", type="success", closable=True, banner=True)

        dialogue_mode = st.selectbox("请选择对话模式：",
                                     ["LLM 对话",
                                      "知识库问答",
                                      "搜索引擎问答",
                                      ],
                                     index=1,
                                     on_change=on_mode_change,
                                     key="dialogue_mode",
                                     )

        def on_llm_change():
            config = get_model_worker_config(llm_model)
            if not config.get("online_api"): # 只有本地model_worker可以切换模型
                st.session_state["prev_llm_model"] = llm_model

        def llm_model_format_func(x):
            if x in running_models:
                return f"{x} (Running)"
            return x

        running_models = api.list_running_models()
        config_models = api.list_config_models()
        for x in running_models:
            if x in config_models:
                config_models.remove(x)
        llm_models = running_models + config_models
        cur_model = st.session_state.get("cur_llm_model", LLM_MODEL)
        index = llm_models.index(cur_model)
        llm_model = st.selectbox("选择LLM模型：",
                                llm_models,
                                index,
                                format_func=llm_model_format_func,
                                on_change=on_llm_change,
                                # key="llm_model",
                                )
        if (st.session_state.get("prev_llm_model") != llm_model
            and not get_model_worker_config(llm_model).get("online_api")):
            with st.spinner(f"正在加载模型： {llm_model}，请勿进行操作或刷新页面"):
                r = api.change_llm_model(st.session_state.get("prev_llm_model"), llm_model)
        st.session_state["cur_llm_model"] = llm_model

        temperature = st.slider("Temperature：", 0.0, 1.0, TEMPERATURE, 0.05)
        history_len = st.number_input("历史对话轮数：", 0, 10, HISTORY_LEN)

        def on_kb_change():
            st.toast(f"已加载知识库： {st.session_state.selected_kb}")

        if dialogue_mode == "知识库问答":
            with st.expander("知识库配置", True):
                kb_list = api.list_knowledge_bases(no_remote_api=True)
                selected_kb = st.selectbox(
                    "请选择知识库：",
                    kb_list,
                    on_change=on_kb_change,
                    key="selected_kb",
                )
                kb_top_k = st.number_input("匹配知识条数：", 1, 20, VECTOR_SEARCH_TOP_K)
                score_threshold = st.slider("知识匹配分数阈值：", 0.0, 1.0, float(SCORE_THRESHOLD), 0.01)
        elif dialogue_mode == "搜索引擎问答":
            search_engine_list = list(SEARCH_ENGINES.keys())
            with st.expander("搜索引擎配置", True):
                search_engine = st.selectbox(
                    label="请选择搜索引擎",
                    options=search_engine_list,
                    index=search_engine_list.index("duckduckgo") if "duckduckgo" in search_engine_list else 0,
                )
                se_top_k = st.number_input("匹配搜索结果条数：", 1, 20, SEARCH_ENGINE_TOP_K)

    # Display chat messages from history on app rerun
    st.file_uploader("在此处上传要描述的图片")
    chat_box.output_messages()
    chat_input_placeholder = "想办法输入语音"
    # chat_input_placeholder2 = webrtc_streamer(
    #     key="sendonly-audio",
    #     mode=WebRtcMode.SENDONLY,
    #     audio_receiver_size=256,
    #     media_stream_constraints={"audio": True},
    # )




# ##############下方为音频识别代码
#     while True:
#         if chat_input_placeholder2.audio_receiver:
#             try:
#                 audio_frames = chat_input_placeholder2.audio_receiver.get_frames(timeout=1)
#             except queue.Empty:
#                 logger.warning("Queue is empty. Abort.")
#                 break
#
#             sound_chunk = pydub.AudioSegment.empty()
#             for audio_frame in audio_frames:
#                 sound = pydub.AudioSegment(
#                     data=audio_frame.to_ndarray().tobytes(),
#                     sample_width=audio_frame.format.bytes,
#                     frame_rate=audio_frame.sample_rate,
#                     channels=len(audio_frame.layout.channels),
#                 )
#                 sound_chunk += sound





#####################以上为音频识别代码



    if prompt := st.chat_input(chat_input_placeholder, key="prompt"):
        history = get_messages_history(history_len)
        # responses = {
        #     "我心情有点不好，你可以安慰我一下吗？":"和我分享一下你的难过吧，或许会好受很多哦。",
        #     "我考试没考好，明明付出了很多努力呀":
        # "我能明白您现在的感受，考试没考好的时候，我们都会感到失望和难过。考试成绩只是学习过程中的一部分，不应该成为定义价值和努力程度的唯一标准，未来还有很多机会去证明自己的能力。重要的是，自己要相信自己，坚持不懈地努力，追求自己的目标。不要把情绪一直压在心头。适当地放松和休息，有助于调整自己的状态和情绪。",
        #     "谢谢你，我感觉好多了~":"我真的很开心能帮助到你，如果你有其他问题，我随时都在这陪着你哦。"
        # }
        # emotion = {
        #     "我心情有点不好，你可以安慰我一下吗？": "|沮丧|",
        #     "我考试没考好，明明付出了很多努力呀":"|疑惑|",
        #     "谢谢你，我感觉好多了~": "|开心|"
        # }
        chat_box.user_say(prompt)
        if dialogue_mode == "LLM 对话":
            chat_box.ai_say("正在思考...")
            # time.sleep(2)
            text = ""
            # response = responses[prompt]
            # for t in iter(response):
            #     if error_msg := check_error_msg(t): # check whether error occured
            #         st.error(error_msg)
            #         break
            #     text += t
            #     time.sleep(0.1)
            #     chat_box.update_msg(text)
            r = api.chat_chat(prompt, history=history, model=llm_model, temperature=temperature)
            for t in r:
                if error_msg := check_error_msg(t): # check whether error occured
                    st.error(error_msg)
                    break
                text += t
                chat_box.update_msg(text)
            chat_box.update_msg(text, streaming=False)  # 更新最终的字符串，去除光标
            sample_rate, audio = get_audio_from_text(model, hps, text)
            st.audio(audio, sample_rate=sample_rate)

        elif dialogue_mode == "知识库问答":
            history = get_messages_history(history_len)
            chat_box.ai_say([
                f"正在查询知识库 `{selected_kb}` ...",
                Markdown("...", in_expander=True, title="知识库匹配结果"),
            ])
            text = ""
            for d in api.knowledge_base_chat(prompt,
                                             knowledge_base_name=selected_kb,
                                             top_k=kb_top_k,
                                             score_threshold=score_threshold,
                                             history=history,
                                             model=llm_model,
                                             temperature=temperature):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                elif chunk := d.get("answer"):
                    text += chunk
                    chat_box.update_msg(text, 0)
            chat_box.update_msg(text, 0, streaming=False)
            chat_box.update_msg("\n\n".join(d.get("docs", [])), 1, streaming=False)
            sample_rate, audio = get_audio_from_text(model, hps, text)
            st.audio(audio, sample_rate=sample_rate)
        elif dialogue_mode == "搜索引擎问答":
            chat_box.ai_say([
                f"正在执行 `{search_engine}` 搜索...",
                Markdown("...", in_expander=True, title="网络搜索结果"),
            ])
            text = ""
            for d in api.search_engine_chat(prompt,
                                            search_engine_name=search_engine,
                                            top_k=se_top_k,
                                            model=llm_model,
                                            temperature=temperature):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                elif chunk := d.get("answer"):
                    text += chunk
                    chat_box.update_msg(text, 0)
            chat_box.update_msg(text, 0, streaming=False)
            chat_box.update_msg("\n\n".join(d.get("docs", [])), 1, streaming=False)
            sample_rate, audio = get_audio_from_text(model, hps, text)
            st.audio(audio, sample_rate=sample_rate)

    now = datetime.now()
    with st.sidebar:

        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            chat_box.reset_history()
            st.experimental_rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )
