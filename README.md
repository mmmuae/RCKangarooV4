(c) 2024, RetiredCoder (RC)

RCKangaroo is free and open-source (GPLv3).
This software demonstrates efficient GPU implementation of SOTA Kangaroo method for solving ECDLP. 
It's part #3 of my research, you can find more details here: https://github.com/RetiredC

Discussion thread: https://bitcointalk.org/index.php?topic=5517607

<b>Features:</b>

- Lowest K=1.15, it means 1.8 times less required operations compared to classic method with K=2.1, also it means that you need 1.8 times less memory to store DPs.
- Fast, about 8GKeys/s on RTX 4090, 4GKeys/s on RTX 3090.
- Keeps DP overhead as small as possible.
- Supports ranges up to 170 bits.
- Both Windows and Linux are supported.

<b>Linux build:</b>

This version is pure SASS at runtime:
- no `nvcc` compile step,
- no CUDA runtime kernel fallback,
- prebuilt cubins are loaded through CUDA Driver API.

You still need CUDA headers/libraries for host code (`cuda_runtime.h`, `libcudart`, `libcuda`).
The Makefile auto-detects CUDA from `CUDA_PATH` or `CUDA_HOME`.
Apple Silicon without NVIDIA/CUDA is not supported for execution.

Examples:

```bash
make all
```

```bash
CUDA_PATH=/usr/local/cuda-13.0 make all
```

<b>Limitations:</b>

- No advanced features like networking, saving/loading DPs, etc.

<b>Command line parameters:</b>

<b>-gpu</b>		which GPUs are used, for example, "035" means that GPUs #0, #3 and #5 are used. If not specified, all available GPUs are used. 

<b>-pubkey</b>		public key to solve, both compressed and uncompressed keys are supported. If not specified, software starts in benchmark mode and solves random keys. 

<b>-start</b>		start offset of the key, in hex. Mandatory if "-pubkey" option is specified. For example, for puzzle #85 start offset is "1000000000000000000000". 

<b>-range</b>		bit range of private the key. Mandatory if "-pubkey" option is specified. For example, for puzzle #85 bit range is "84" (84 bits). Must be in range 32...170. 

<b>-dp</b>		DP bits. Must be in range 14...60. Low DP bits values cause larger DB but reduces DP overhead and vice versa. 

<b>-max</b>		option to limit max number of operations. For example, value 5.5 limits number of operations to 5.5 * 1.15 * sqrt(range), software stops when the limit is reached. 

<b>-tames</b>		filename with tames. If file not found, software generates tames (option "-max" is required) and saves them to the file. If the file is found, software loads tames to speedup solving. 

<b>-dpf-mode</b>		run worker in DP export mode (no local key solving path). Values: <code>wild</code>, <code>tame</code>, <code>both</code>.

<b>-dpf-dir</b>		required with DP export mode. Local directory where `.wdp` chunks are written.

<b>-dpf-worker</b>		required with DP export mode. Worker identifier stored in every `.wdp` chunk.

<b>-dpf-session</b>		optional session tag for `.wdp` filenames and metadata. Auto-generated if omitted.

<b>-dpf-flush-records</b>	optional chunk size in DP records for `.wdp` output (default 1000000).

<b>-dpf-flush-sec</b>	optional max seconds before flushing a partial `.wdp` chunk (default 10).

<b>-h</b>, <b>--help</b>		show CLI help.

If <code>-pubkey</code> is not provided, the app runs in benchmark mode by design.

SASS backend files (required):
<code>sass/sm89/rckangaroo_kernels.cubin</code> and <code>sass/sm120/rckangaroo_kernels.cubin</code>.
You can also set <code>RCK_SASS_DIR</code> to point to another base folder with the same structure.
Advanced SASS runtime envs:
<code>RCK_SASS_CUBIN</code> (direct cubin path),
<code>RCK_SASS_VARIANT</code> (load <code>rckangaroo_kernels_&lt;variant&gt;.cubin</code> before default),
<code>RCK_GROUP_CNT</code> and <code>RCK_BLOCKCNT_MUL</code> for launch profile tuning.
For SM120/RTX 5090, default runtime launch profile uses group count 64.

Benchmark script:
<code>scripts/bench_seq.sh</code> and <code>scripts/run_main.sh</code>.
Examples:
<code>./scripts/bench_seq.sh 78 16 12 3 ./rckangaroo</code>,
<code>./scripts/run_main.sh 03... 1000000000000000000000 84 16 ./rckangaroo</code>.

Compatibility aliases:
<code>-dp-export</code> -> <code>-dpf-mode</code>,
<code>-wild-only</code> -> <code>-dpf-mode wild</code>,
<code>-wild-spool-dir</code> -> <code>-dpf-dir</code>,
<code>-worker-id</code> -> <code>-dpf-worker</code>,
<code>-session-tag</code> -> <code>-dpf-session</code>,
<code>-wild-flush-records</code> -> <code>-dpf-flush-records</code>,
<code>-wild-flush-sec</code> -> <code>-dpf-flush-sec</code>.

When public key is solved, software displays it and also writes it to "RESULTS.TXT" file. 

Sample command line for puzzle #85:

RCKangaroo.exe -dp 16 -range 84 -start 1000000000000000000000 -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a

Equivalent Linux invocation:

./rckangaroo -dp 16 -range 84 -start 1000000000000000000000 -pubkey 0329c4574a4fd8c810b7e42a4b398882b381bcd85e40c6883712912d167c83e73a

Sample command to generate tames:

RCKangaroo.exe -dp 16 -range 76 -tames tames76.dat -max 10

Then you can restart software with same parameters to see less K in benchmark mode or add "-tames tames76.dat" to solve some public key in 76-bit range faster.

Sample command for DP export (wild):

RCKangaroo -dpf-mode wild -range 134 -dp 30 -start 0 -pubkey 03... -dpf-worker worker01 -dpf-dir /tmp/wildspool

<b>Some notes:</b>

Fastest ECDLP solvers will always use SOTA/SOTA+ method, as it's 1.4/1.5 times faster and requires less memory for DPs compared to the best 3-way kangaroos with K=1.6. 
Even if you already have a faster implementation of kangaroo jumps, incorporating SOTA method will improve it further. 
While adding the necessary loop-handling code will cause you to lose about 5–15% of your current speed, the SOTA method itself will provide a 40% performance increase. 
Overall, this translates to roughly a 25% net improvement, which should not be ignored if your goal is to build a truly fast solver. 


<b>Changelog:</b>

v3.1:

- fixed "gpu illegal memory access" bug.
- some small improvements.

v3.0:

- added "-tames" and "-max" options.
- fixed some bugs.

v2.0:

- added support for 30xx, 20xx and 1xxx cards.
- some minor changes.

v1.1:

- added ability to start software on 30xx cards.

v1.0:

- initial release.
