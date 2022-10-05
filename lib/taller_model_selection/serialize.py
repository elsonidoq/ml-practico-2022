import json


# Funciones utiles para escribir y leer un archivo en formato json lines

def write_jl(docs, fname):
    with open(fname, 'w') as f:
        for i, doc in enumerate(docs):
            if i > 0: f.write('\n')
            f.write(json.dumps(doc))


def iter_jl(fname):
    with open(fname) as f:
        for line in f:
            yield json.loads(line)
