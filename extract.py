# coding: utf-8

import json
import re
import sys

# This is not the prefect way to match emojis. But just use it as a demo.
re_emoji = re.compile(u'['
    u'\U0001F300-\U0001F64F'
    u'\U0001F680-\U0001F6FF'
    u'\u2600-\u26FF\u2700-\u27BF]', 
    re.UNICODE)

c_bad_json = 0
c_bad_lang = 0
c_no_text = 0

for line in sys.stdin:
    try:
        t = json.loads(line)
    except:
        c_bad_json += 1
        continue
    if 'lang' not in t or t['lang'] != 'en':
        c_bad_lang += 1
        continue
    if 'text' not in t:
        c_no_text += 1
        continue
    text = t['text']
    if re_emoji.search(text):
        print(line.strip())
