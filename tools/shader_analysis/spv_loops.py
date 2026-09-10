"""Find loops that can never exit in dumped SPIR-V.

spirv-val accepts an infinite loop: it is structurally legal. On a GPU it is a
hang, and a hang past the Windows TDR window is a device loss. This walks each
function's CFG and reports any OpLoopMerge whose merge block nothing inside the
loop branches to.
"""
import sys, struct, os, glob
from collections import defaultdict

MAGIC = 0x07230203
OP_FUNCTION, OP_FUNCTION_END = 54, 56
OP_LOOP_MERGE, OP_SELECTION_MERGE, OP_LABEL = 246, 247, 248
OP_BRANCH, OP_BRANCH_COND, OP_SWITCH = 249, 250, 251
OP_KILL, OP_RETURN, OP_RETURN_VALUE, OP_UNREACHABLE = 252, 253, 254, 255
OP_TERMINATE_INVOCATION, OP_DEMOTE = 4416, 5380

TERMINATORS = {OP_KILL, OP_RETURN, OP_RETURN_VALUE, OP_UNREACHABLE,
               OP_TERMINATE_INVOCATION, OP_DEMOTE}


def parse(path):
    data = open(path, 'rb').read()
    if len(data) < 20:
        return None
    magic = struct.unpack('<I', data[:4])[0]
    endian = '<'
    if magic != MAGIC:
        magic_be = struct.unpack('>I', data[:4])[0]
        if magic_be != MAGIC:
            return None
        endian = '>'
    n = len(data) // 4
    words = struct.unpack(f'{endian}{n}I', data[:n * 4])
    insts, i = [], 5
    while i < n:
        w = words[i]
        count, op = w >> 16, w & 0xFFFF
        if count == 0 or i + count > n:
            break
        insts.append((op, words[i + 1:i + count]))
        i += count
    return insts


def functions(insts):
    """Yield (func_id, blocks, loops). blocks: label -> [successors]."""
    cur = None
    for op, args in insts:
        if op == OP_FUNCTION:
            cur = {'id': args[1] if len(args) > 1 else 0,
                   'blocks': {}, 'loops': [], 'order': []}
            block = None
            pending_loop = None
        elif cur is None:
            continue
        elif op == OP_FUNCTION_END:
            yield cur
            cur = None
        elif op == OP_LABEL:
            block = args[0]
            cur['blocks'][block] = []
            cur['order'].append(block)
            pending_loop = None
        elif block is None:
            continue
        elif op == OP_LOOP_MERGE:
            pending_loop = (block, args[0], args[1])  # header, merge, continue
            cur['loops'].append(pending_loop)
        elif op == OP_BRANCH:
            cur['blocks'][block].append(args[0])
        elif op == OP_BRANCH_COND:
            cur['blocks'][block].extend([args[1], args[2]])
        elif op == OP_SWITCH:
            cur['blocks'][block].append(args[1])
            rest = args[2:]
            # (literal..., label) pairs; literal width is 1 word for 32-bit selectors
            for k in range(0, len(rest) - 1, 2):
                cur['blocks'][block].append(rest[k + 1])
        elif op in TERMINATORS:
            pass


def loop_body(blocks, header, merge):
    """Blocks reachable from header without passing through merge."""
    seen, stack = set(), [header]
    while stack:
        b = stack.pop()
        if b in seen or b == merge:
            continue
        seen.add(b)
        for s in blocks.get(b, []):
            if s != merge:
                stack.append(s)
    return seen


def analyze(path):
    insts = parse(path)
    if insts is None:
        return [('PARSE', 'not a SPIR-V module')]
    findings = []
    for fn in functions(insts):
        blocks = fn['blocks']
        reachable = loop_body(blocks, fn['order'][0], None) if fn['order'] else set()
        for header, merge, cont in fn['loops']:
            if header not in reachable:
                continue  # dead code, not executed
            body = loop_body(blocks, header, merge)
            exits = [b for b in body if merge in blocks.get(b, [])]
            if not exits:
                findings.append(
                    ('INFINITE',
                     f'loop header %{header} merge %{merge} continue %{cont}: '
                     f'{len(body)} body blocks, none branch to the merge'))
            if merge not in blocks and merge not in (b for b in blocks):
                findings.append(('MISSING_MERGE', f'loop header %{header} merge %{merge} has no label'))
    return findings


def main():
    targets = []
    for a in sys.argv[1:]:
        targets.extend(glob.glob(a))
    if not targets:
        print('usage: spv_loops.py <files or globs>')
        return 1
    total, bad = 0, 0
    for path in sorted(targets):
        total += 1
        for kind, msg in analyze(path):
            bad += 1
            print(f'{kind:14s} {os.path.basename(path)}: {msg}')
    print(f'\nscanned {total} modules, {bad} findings')
    return 0


if __name__ == '__main__':
    sys.exit(main())
