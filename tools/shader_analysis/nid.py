import hashlib, base64, struct

SUFFIX = bytes([0x51,0x8D,0x64,0xA6,0x35,0xDE,0xD8,0xC1,
                0xE6,0xB0,0x39,0xB1,0xC3,0xE5,0x52,0x30])

def nid(name):
    h = hashlib.sha1(name.encode('utf-8') + SUFFIX).digest()
    n = struct.unpack('<Q', h[:8])[0]
    enc = base64.b64encode(n.to_bytes(8, 'big')).decode()[:11]
    return enc.replace('/', '-')

if __name__ == '__main__':
    # verify the oracle against NIDs already registered in libNet.cpp
    known = {
        'sceHttpInit':            'A9cVMUtEp4Y',
        'sceHttpTerm':            'Ik-KpLTlf7Q',
        'sceHttpCreateTemplate':  '0gYjPTR-6cY',
        'sceHttpCreateConnection':'Kiwv9r4IZCc',
        'sceHttpSendRequest':     '1e2BNwI-XzE',
        'sceHttpGetStatusCode':   '0a2TBNfE3BU',
    }
    ok = True
    for name, expect in known.items():
        got = nid(name)
        flag = 'OK ' if got == expect else 'BAD'
        if got != expect: ok = False
        print(f'{flag} {name:32s} expect={expect} got={got}')
    print('\nORACLE VALID' if ok else '\nORACLE BROKEN')
