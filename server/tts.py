import io
import requests
from server.utils import api_address
api_base_url = api_address()
def text2speech(input_text):
    api="/111test"
    url = f"{api_base_url}{api}"
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json',
}
    input_text="你今天快乐吗"
    data={"text":input_text}
    response=requests.post(url,headers=headers,json=data)

    if response.status_code==200:
        audio_stream=io.BytesIO(response.content)
        audio_stream.name='speech.mp3'
        return audio_stream
        return "success"
    else:
        print("Error:",response.status_code)
        return None