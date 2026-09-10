"""Compare two shader dump folders by shader hash.

Dump filenames are <seq>_new_shader_<stage>_<hash>.spv, so the same guest shader
gets a different sequence number every run but keeps its hash. Group by hash and
report which shaders the recompiler now emits differently, which is exactly the
set a structurizer change is responsible for.

usage: spv_diff.py <old_dir> <new_dir>
"""
import sys, os, glob, re, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spv_loops import parse, functions

NAME = re.compile(r'^\d+_new_shader_([a-z]+)_([0-9a-f]{16})\.spv$')


def collect(folder):
    out = {}
    for path in glob.glob(os.path.join(folder, '*.spv')):
        m = NAME.match(os.path.basename(path))
        if not m:
            continue
        stage, h = m.group(1), m.group(2)
        blob = open(path, 'rb').read()
        # keep one representative per hash; identical hashes recompile identically
        out.setdefault((stage, h), []).append((path, hashlib.sha1(blob).hexdigest(), blob))
    return out


def shape(blob_path):
    insts = parse(blob_path)
    if insts is None:
        return None
    blocks = loops = 0
    for fn in functions(insts):
        blocks += len(fn['blocks'])
        loops += len(fn['loops'])
    switches = sum(1 for op, _ in insts if op == 251)
    sels = sum(1 for op, _ in insts if op == 247)
    return blocks, loops, sels, switches


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    old, new = collect(sys.argv[1]), collect(sys.argv[2])
    old_keys, new_keys = set(old), set(new)

    changed, same = [], 0
    for key in sorted(old_keys & new_keys):
        o_sha = old[key][0][1]
        n_sha = new[key][0][1]
        if o_sha == n_sha:
            same += 1
            continue
        changed.append((key, old[key][0][0], new[key][0][0]))

    print(f'shaders in both dumps : {len(old_keys & new_keys)}')
    print(f'  byte-identical      : {same}')
    print(f'  CHANGED             : {len(changed)}')
    print(f'only in old dump      : {len(old_keys - new_keys)}')
    print(f'only in new dump      : {len(new_keys - old_keys)}')

    if changed:
        print(f'\n{"stage":5s} {"hash":16s}  {"blocks":>13s} {"loops":>9s} '
              f'{"selects":>13s} {"switch":>9s}')
        for (stage, h), opath, npath in changed:
            o, n = shape(opath), shape(npath)
            if o is None or n is None:
                print(f'{stage:5s} {h}  <unparsed>')
                continue
            def d(a, b):
                return f'{a}->{b}' if a != b else str(a)
            print(f'{stage:5s} {h}  {d(o[0],n[0]):>13s} {d(o[1],n[1]):>9s} '
                  f'{d(o[2],n[2]):>13s} {d(o[3],n[3]):>9s}')
        print('\nA shader that lost its OpSwitch left the dispatcher fallback.')
        print('Run spv_loops.py over the new files above before anything else.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
