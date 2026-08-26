# Dump scalar GGUF metadata keys from a GGUF file (debug tool)
import struct, sys

path = sys.argv[1] if len(sys.argv) > 1 else 'models/Qwen3-8B-Q4_K_M.gguf'

with open(path, 'rb') as f:
    assert f.read(4) == b'GGUF', 'not a GGUF file'
    ver = struct.unpack('<I', f.read(4))[0]
    tc, kvc = struct.unpack('<QQ', f.read(16))
    print(f'version={ver} tensor_count={tc} kv_count={kvc}')

    def rs():
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8', 'replace')

    fmts = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',10:'<Q',11:'<q',12:'<d'}

    def skip(t):
        if t in fmts:
            f.seek(struct.calcsize(fmts[t]), 1)
        elif t == 7:
            f.seek(1, 1)
        elif t == 8:
            rs()
        elif t == 9:
            et = struct.unpack('<I', f.read(4))[0]
            n = struct.unpack('<Q', f.read(8))[0]
            for _ in range(min(n, 10_000_000)):
                skip(et)

    for _ in range(kvc):
        k = rs()
        t = struct.unpack('<I', f.read(4))[0]
        v = None
        if t in fmts:
            v = struct.unpack(fmts[t], f.read(struct.calcsize(fmts[t])))[0]
        elif t == 7:
            v = bool(f.read(1)[0])
        elif t == 8:
            v = rs()
        else:
            skip(t)
            continue
        print(f'{k} = {v}')
