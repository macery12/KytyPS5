# Shader analysis tools

Offline analysis for the shader recompiler. These exist so a recompiler problem can be
investigated by reading dumps rather than by repeatedly running a game and waiting for a GPU
crash. Plain Python 3, no dependencies.

## Getting the dumps

Run the emulator with `--shader-log-direction File` and a `--shader-log-folder`. Every generated
shader is written as `<seq>_new_shader_<stage>_<hash>.spv`, with the original AGC bytecode and
its disassembly under `original/`. This costs about 59 MB for a boot, unlike
`--graphics-debug-dump`, which costs hundreds of megabytes and slows startup enough to change
what reproduces.

## spv_loops.py

```sh
python spv_loops.py "_Shaders/*.spv"
```

Reports any `OpLoopMerge` whose merge block nothing inside the loop branches to.

`spirv-val` proves a module is *legal*; it says nothing about whether it *terminates*. An
infinite loop is perfectly valid SPIR-V and, on a GPU, a hang - and a hang past the Windows TDR
window is a device loss reported as `VK_ERROR_DEVICE_LOST` on some later unrelated submit. This
is the check that catches a structurizer emitting a loop that cannot exit.

## spv_diff.py

```sh
python spv_diff.py _Shaders _ShadersNew
```

Groups both folders by shader hash and reports which shaders the recompiler now emits
differently. Dump sequence numbers change every run; hashes do not.

After a structurizer change this is the fastest way to get the blast radius. The set of shaders
whose SPIR-V changed is exactly the set the change is responsible for, and it is usually small
enough to read by hand. A shader that lost its `OpSwitch` left the dispatcher fallback.

## nid.py

```sh
python -c "from nid import nid; print(nid('sceHttpGetLastErrno'))"
```

Computes a PS5 NID: SHA-1 of the symbol name plus the standard 16-byte suffix key, first 8 bytes
read little-endian, re-emitted big-endian, base64 with `/` mapped to `-`.

For an unresolved import, generate candidate names for the module and match their NIDs against
the unknown one. Eleven NHL 26 imports were identified this way. Always validate the oracle
against NIDs already registered in the relevant `LIB_DEFINE` block before trusting a result.
