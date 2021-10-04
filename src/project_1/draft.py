import json

with open('test.tsv', 'r') as f:
    lines = f.readlines()
    f.close()
    d = {k: v for k, v in map(lambda x: x.split('\t'), lines)}
    with open('test.json', 'w') as j:
        j.write(json.dumps(d))
        j.close()
