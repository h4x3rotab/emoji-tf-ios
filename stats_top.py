# coding: utf-8
import re
import json
from collections import defaultdict

# This is not the prefect way to match emojis. But just use it as a demo.
re_emoji = re.compile(u'['
    u'\U0001F300-\U0001F64F'
    u'\U0001F680-\U0001F6FF'
    u'\u2600-\u26FF\u2700-\u27BF]', 
    re.UNICODE)

c = defaultdict(int)
readed = 0
last_reported = 0

try:
    with open('data/extracted.list') as fin:
        for line in fin:
            readed += len(line)
            if not line.strip():
                continue
            t = json.loads(line)
            for emoji in re_emoji.findall(t['text']):
                c[emoji] += 1

            readed_mb = readed // (100 * 1024 * 1024)
            if readed_mb > last_reported:
                last_reported = readed_mb
                print('Processed:', readed_mb * 100)
finally:
    out = ''
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        out += '{}: {}\n'.format(k, v)
    print(out)
    with open('data/stat.txt', 'w') as fout:
        fout.write(out)
