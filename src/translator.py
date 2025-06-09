# src/translator.py

import requests
from config import TRANSLATION_SOURCE_LANG, TRANSLATION_TARGET_LANG

def translate_text(text):
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": TRANSLATION_SOURCE_LANG,
            "tl": TRANSLATION_TARGET_LANG,
            "dt": "t",
            "q": text
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()[0][0][0]
        return "[Translation Error]"
    except Exception as e:
        return "[Error]"
