// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <cstdarg>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cctype>
#include <cstring>
#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "cuda.h"

#include "GpuKang.h"

void AddPointsToList(u32* data, int cnt, u64 ops_cnt);
extern bool gGenMode; //tames generation mode
extern u32 gDpExportMode;

static bool FileExists(const char* path)
{
	FILE* fp = fopen(path, "rb");
	if (!fp)
		return false;
	fclose(fp);
	return true;
}

static bool TryBuildCubinPath(const char* base_dir, const char* arch_tag, const char* variant, char* out_path, size_t out_size)
{
	if (!base_dir || !base_dir[0] || !arch_tag || !arch_tag[0] || !out_path || !out_size)
		return false;
	if (variant && variant[0])
	{
		snprintf(out_path, out_size, "%s/%s/rckangaroo_kernels_%s.cubin", base_dir, arch_tag, variant);
		if (FileExists(out_path))
			return true;
	}
	snprintf(out_path, out_size, "%s/%s/rckangaroo_kernels.cubin", base_dir, arch_tag);
	return FileExists(out_path);
}

static bool GetSassArchTag(int major, int minor, char* arch_tag, size_t arch_tag_size)
{
	if (!arch_tag || !arch_tag_size)
		return false;
	arch_tag[0] = 0;
	if ((major == 8) && (minor == 9))
	{
		snprintf(arch_tag, arch_tag_size, "sm89");
		return true;
	}
	if ((major == 12) && (minor == 0))
	{
		snprintf(arch_tag, arch_tag_size, "sm120");
		return true;
	}
	return false;
}

static bool BuildSassCubinPath(int major, int minor, char* out_path, size_t out_size)
{
	if (!out_path || !out_size)
		return false;
	out_path[0] = 0;

	char arch_tag[16];
	if (!GetSassArchTag(major, minor, arch_tag, sizeof(arch_tag)))
		return false;

	const char* direct_cubin = getenv("RCK_SASS_CUBIN");
	if (direct_cubin && direct_cubin[0])
	{
		snprintf(out_path, out_size, "%s", direct_cubin);
		return FileExists(out_path);
	}

	const char* variant = getenv("RCK_SASS_VARIANT");

	const char* env_dir = getenv("RCK_SASS_DIR");
	if (TryBuildCubinPath(env_dir, arch_tag, variant, out_path, out_size))
		return true;
	if (TryBuildCubinPath("sass", arch_tag, variant, out_path, out_size))
		return true;

	return false;
}

static void FormatCuError(CUresult rc, char* out_buf, size_t out_size)
{
	if (!out_buf || !out_size)
		return;
	out_buf[0] = 0;
	const char* err_name = nullptr;
	const char* err_text = nullptr;
	cuGetErrorName(rc, &err_name);
	cuGetErrorString(rc, &err_text);
	if (!err_name)
		err_name = "UNKNOWN";
	if (!err_text)
		err_text = "no error text";
	snprintf(out_buf, out_size, "%s (%s)", err_name, err_text);
}

static int ReadEnvIntBounded(const char* name, int fallback, int min_val, int max_val)
{
	const char* value = getenv(name);
	if (!value || !value[0])
		return fallback;
	char* end_ptr = nullptr;
	long parsed = strtol(value, &end_ptr, 10);
	if (!end_ptr || (*end_ptr != 0))
		return fallback;
	if (parsed < min_val)
		return fallback;
	if (parsed > max_val)
		return fallback;
	return (int)parsed;
}

static bool IsEnvSet(const char* name)
{
	const char* value = getenv(name);
	return value && value[0];
}

static bool ReadEnvBool(const char* name, bool fallback)
{
	const char* value = getenv(name);
	if (!value || !value[0])
		return fallback;
	auto EqNoCase = [](const char* a, const char* b) -> bool
	{
		if (!a || !b)
			return false;
		while (*a && *b)
		{
			if (tolower((unsigned char)*a) != tolower((unsigned char)*b))
				return false;
			a++;
			b++;
		}
		return (*a == 0) && (*b == 0);
	};
	if ((strcmp(value, "0") == 0) || EqNoCase(value, "false") || EqNoCase(value, "off") || EqNoCase(value, "no"))
		return false;
	if ((strcmp(value, "1") == 0) || EqNoCase(value, "true") || EqNoCase(value, "on") || EqNoCase(value, "yes"))
		return true;
	return fallback;
}

static const char* PathBasename(const char* path)
{
	if (!path)
		return "";
	const char* slash = strrchr(path, '/');
	const char* backslash = strrchr(path, '\\');
	const char* p = slash;
	if (!p || (backslash && backslash > p))
		p = backslash;
	return p ? (p + 1) : path;
}

static bool TryParseGroupHint(const char* text, u32& out_group)
{
	if (!text || !text[0])
		return false;
	for (size_t i = 0; text[i]; i++)
	{
		if ((text[i] != 'g') && (text[i] != 'G'))
			continue;
		size_t j = i + 1;
		if (!isdigit((unsigned char)text[j]))
			continue;
		u32 val = 0;
		while (isdigit((unsigned char)text[j]))
		{
			val = val * 10u + (u32)(text[j] - '0');
			j++;
		}
		char term = text[j];
		if (term && (term != '.') && (term != '_') && (term != '-') && (term != '/') && (term != '\\'))
			continue;
		if ((val >= 8u) && (val <= 128u) && ((val % 8u) == 0u))
		{
			out_group = val;
			return true;
		}
	}
	return false;
}

static bool TryParseBlockHint(const char* text, u32& out_block)
{
	if (!text || !text[0])
		return false;
	for (size_t i = 0; text[i]; i++)
	{
		if ((text[i] != 'b') && (text[i] != 'B'))
			continue;
		size_t j = i + 1;
		if (!isdigit((unsigned char)text[j]))
			continue;
		u32 val = 0;
		while (isdigit((unsigned char)text[j]))
		{
			val = val * 10u + (u32)(text[j] - '0');
			j++;
		}
		char term = text[j];
		if (term && (term != '.') && (term != '_') && (term != '-') && (term != '/') && (term != '\\'))
			continue;
		if ((val >= 64u) && (val <= 1024u) && ((val % 32u) == 0u))
		{
			out_block = val;
			return true;
		}
	}
	return false;
}

static u32 DefaultGroupCntByArch(bool isOldGpu, int sm_major, int sm_minor)
{
	if (isOldGpu)
		return PNT_GROUP_OLD_GPU;
	// Hopper/Blackwell profile for pure SASS in this branch.
	if ((sm_major >= 12) && (sm_minor >= 0))
		return 64;
	if ((sm_major == 8) && (sm_minor == 9))
		return 24;
	return PNT_GROUP_NEW_GPU;
}

static double MinDpsPerKangTargetByRange(int rangeBits)
{
	if (rangeBits <= 80)
		return 9.0;
	if (rangeBits <= 96)
		return 7.0;
	if (rangeBits <= 112)
		return 5.0;
	if (rangeBits <= 128)
		return 4.0;
	if (rangeBits <= 144)
		return 3.0;
	return 2.0;
}

static void TuneGroupCntForMainSolve(u32& groupCnt, u32 blockCnt, u32 blockSize, int rangeBits, int dpBits, int cudaIndex)
{
	if ((groupCnt < 8) || (rangeBits < 32) || (dpBits < 1) || (dpBits > 60))
		return;

	const double minDpsPerKang = MinDpsPerKangTargetByRange(rangeBits);
	const double ops = 1.15 * pow(2.0, rangeBits / 2.0);
	const double dpValue = ldexp(1.0, dpBits);
	u32 tuned = groupCnt;
	while (tuned > 8)
	{
		double kangCnt = (double)blockCnt * (double)blockSize * (double)tuned;
		double dpsPerKang = ops / (kangCnt * dpValue);
		if (dpsPerKang >= minDpsPerKang)
			break;
		tuned -= 8;
	}
	if (tuned != groupCnt)
	{
		printf("GPU %d: auto-tuned group count %u -> %u for stable collisions (range=%d, dp=%d, target DPs/kang>=%.1f)\r\n",
			cudaIndex,
			groupCnt,
			tuned,
			rangeBits,
			dpBits,
			minDpsPerKang);
		groupCnt = tuned;
	}
}

int RCGpuKang::CalcKangCnt(int rangeBits, int dpBits, u32 runMode)
{
	ResolveLaunchConfig(Kparams.BlockCnt, Kparams.BlockSize, Kparams.GroupCnt);
	if ((runMode == KANG_MODE_MAIN) && !IsEnvSet("RCK_GROUP_CNT") && !LaunchGroupLocked)
		TuneGroupCntForMainSolve(Kparams.GroupCnt, Kparams.BlockCnt, Kparams.BlockSize, rangeBits, dpBits, CudaIndex);
	return Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
}

void RCGpuKang::SetBackendError(const char* fmt, ...)
{
	if (!fmt)
	{
		BackendError[0] = 0;
		return;
	}
	va_list args;
	va_start(args, fmt);
	vsnprintf(BackendError, sizeof(BackendError), fmt, args);
	va_end(args);
	BackendError[sizeof(BackendError) - 1] = 0;
}

void RCGpuKang::ResolveLaunchConfig(u32& blockCnt, u32& blockSize, u32& groupCnt)
{
	blockSize = IsOldGpu ? BLOCK_SIZE_OLD_GPU : BLOCK_SIZE_NEW_GPU;
	LaunchConfigError[0] = 0;
	LaunchConfigSource[0] = 0;
	LaunchStrictMode = ReadEnvBool("RCK_SASS_STRICT", true);
	LaunchGroupLocked = false;
	LaunchContractBlockSize = blockSize;
	LaunchContractGroupCnt = DefaultGroupCntByArch(IsOldGpu, SmMajor, SmMinor);
	blockSize = LaunchContractBlockSize;
	groupCnt = LaunchContractGroupCnt;

	const char* variant = getenv("RCK_SASS_VARIANT");
	u32 hinted_group = 0;
	u32 hinted_block = 0;
	const bool has_group_hint = TryParseGroupHint(variant, hinted_group);
	const bool has_block_hint = TryParseBlockHint(variant, hinted_block);
	if (has_group_hint || has_block_hint)
	{
		if (has_group_hint)
		{
			LaunchContractGroupCnt = hinted_group;
			groupCnt = hinted_group;
		}
		if (has_block_hint)
		{
			LaunchContractBlockSize = hinted_block;
			blockSize = hinted_block;
		}
		snprintf(LaunchConfigSource, sizeof(LaunchConfigSource), "variant=%s", variant);
		LaunchGroupLocked = true;
	}
	else
	{
		const char* direct_cubin = getenv("RCK_SASS_CUBIN");
		if (direct_cubin && direct_cubin[0])
		{
			const char* cubin_name = PathBasename(direct_cubin);
			const bool has_cubin_group = TryParseGroupHint(cubin_name, hinted_group);
			const bool has_cubin_block = TryParseBlockHint(cubin_name, hinted_block);
			if (has_cubin_group || has_cubin_block)
			{
				if (has_cubin_group)
				{
					LaunchContractGroupCnt = hinted_group;
					groupCnt = hinted_group;
				}
				if (has_cubin_block)
				{
					LaunchContractBlockSize = hinted_block;
					blockSize = hinted_block;
				}
				snprintf(LaunchConfigSource, sizeof(LaunchConfigSource), "cubin=%s", cubin_name);
				LaunchGroupLocked = true;
			}
		}
	}

	if (!LaunchConfigSource[0])
		snprintf(LaunchConfigSource, sizeof(LaunchConfigSource), "arch_default");

	// Optional env overrides for quick tuning without CLI churn.
	bool env_group_set = IsEnvSet("RCK_GROUP_CNT");
	int envGroupCnt = ReadEnvIntBounded("RCK_GROUP_CNT", groupCnt, 8, 64);
	envGroupCnt = (envGroupCnt / 8) * 8;
	if (envGroupCnt < 8)
		envGroupCnt = 8;
	if (env_group_set)
	{
		if (LaunchStrictMode && (u32)envGroupCnt != LaunchContractGroupCnt)
		{
			snprintf(
				LaunchConfigError,
				sizeof(LaunchConfigError),
				"RCK_GROUP_CNT=%d mismatches SASS contract group=%u (%s). Use matching cubin/profile or set RCK_SASS_STRICT=0.",
				envGroupCnt,
				LaunchContractGroupCnt,
				LaunchConfigSource);
		}
		groupCnt = (u32)envGroupCnt;
	}

	bool env_block_set = IsEnvSet("RCK_BLOCK_SIZE");
	int envBlockSize = ReadEnvIntBounded("RCK_BLOCK_SIZE", (int)blockSize, 64, 1024);
	envBlockSize = (envBlockSize / 32) * 32;
	if (envBlockSize < 64)
		envBlockSize = 64;
	if (env_block_set)
	{
		if (LaunchStrictMode && (u32)envBlockSize != LaunchContractBlockSize)
		{
			snprintf(
				LaunchConfigError,
				sizeof(LaunchConfigError),
				"RCK_BLOCK_SIZE=%d mismatches SASS contract block=%u (%s). Use matching cubin/profile or set RCK_SASS_STRICT=0.",
				envBlockSize,
				LaunchContractBlockSize,
				LaunchConfigSource);
		}
		blockSize = (u32)envBlockSize;
	}

	// In strict mode we lock launch geometry to the cubin contract to prevent
	// silent DP-class regressions from host/cubin desync.
	if (LaunchStrictMode)
	{
		blockSize = LaunchContractBlockSize;
		groupCnt = LaunchContractGroupCnt;
		LaunchGroupLocked = true;
	}

	int blockCntMul = ReadEnvIntBounded("RCK_BLOCKCNT_MUL", 1, 1, 8);
	blockCnt = mpCnt * blockCntMul;
}

bool RCGpuKang::InitSassBackend(const u64* jmp2_table)
{
	SetBackendError(nullptr);
	LoadedSassPath[0] = 0;
	SassModule = nullptr;
	SassKernelA = nullptr;
	SassKernelB = nullptr;
	SassKernelC = nullptr;
	SassKernelGen = nullptr;

	if (!jmp2_table)
	{
		SetBackendError("internal error: null jmp2_table");
		return false;
	}

	cudaDeviceProp dev_prop;
	cudaError_t rt_err = cudaGetDeviceProperties(&dev_prop, CudaIndex);
	if (rt_err != cudaSuccess)
	{
		SetBackendError("cudaGetDeviceProperties failed: %s", cudaGetErrorString(rt_err));
		return false;
	}

	char cubin_path[512];
	if (!BuildSassCubinPath(dev_prop.major, dev_prop.minor, cubin_path, sizeof(cubin_path)))
	{
		SetBackendError("no SASS cubin for sm%d%d (searched RCK_SASS_CUBIN, RCK_SASS_DIR, ./sass)", dev_prop.major, dev_prop.minor);
		return false;
	}
	const char* variant = getenv("RCK_SASS_VARIANT");
	const char* direct_cubin = getenv("RCK_SASS_CUBIN");
	if (LaunchStrictMode && variant && variant[0] && !(direct_cubin && direct_cubin[0]))
	{
		char expected_tail[96];
		snprintf(expected_tail, sizeof(expected_tail), "_%s.cubin", variant);
		if (!strstr(PathBasename(cubin_path), expected_tail))
		{
			SetBackendError(
				"requested variant '%s' not found (loaded %s). Build matching cubin or unset RCK_SASS_VARIANT.",
				variant,
				PathBasename(cubin_path));
			return false;
		}
	}
	u32 path_group_hint = 0;
	u32 path_block_hint = 0;
	const bool has_path_hint = TryParseGroupHint(PathBasename(cubin_path), path_group_hint);
	const bool has_path_block_hint = TryParseBlockHint(PathBasename(cubin_path), path_block_hint);
	if (LaunchStrictMode && has_path_hint && (path_group_hint != Kparams.GroupCnt))
	{
		SetBackendError(
			"launch group=%u mismatches cubin hint group=%u (%s)",
			Kparams.GroupCnt,
			path_group_hint,
			PathBasename(cubin_path));
		return false;
	}
	if (LaunchStrictMode && has_path_block_hint && (path_block_hint != Kparams.BlockSize))
	{
		SetBackendError(
			"launch block=%u mismatches cubin hint block=%u (%s)",
			Kparams.BlockSize,
			path_block_hint,
			PathBasename(cubin_path));
		return false;
	}

	CUresult cu_err = cuInit(0);
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		SetBackendError("cuInit failed: %s", errbuf);
		return false;
	}

	// Ensure runtime primary context exists before Driver API calls.
	cudaFree(0);

	CUcontext ctx = nullptr;
	cu_err = cuCtxGetCurrent(&ctx);
	if ((cu_err != CUDA_SUCCESS) || !ctx)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		SetBackendError("cuCtxGetCurrent failed: %s", errbuf);
		return false;
	}

	cu_err = cuModuleLoad(&SassModule, cubin_path);
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		SetBackendError("cuModuleLoad(%s) failed: %s", cubin_path, errbuf);
		return false;
	}

	cu_err = cuModuleGetFunction(&SassKernelA, SassModule, "KernelA");
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		SetBackendError("cuModuleGetFunction(KernelA) failed: %s", errbuf);
		ReleaseBackend();
		return false;
	}
	cu_err = cuModuleGetFunction(&SassKernelB, SassModule, "KernelB");
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		SetBackendError("cuModuleGetFunction(KernelB) failed: %s", errbuf);
		ReleaseBackend();
		return false;
	}
	cu_err = cuModuleGetFunction(&SassKernelC, SassModule, "KernelC");
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		SetBackendError("cuModuleGetFunction(KernelC) failed: %s", errbuf);
		ReleaseBackend();
		return false;
	}
	cu_err = cuModuleGetFunction(&SassKernelGen, SassModule, "KernelGen");
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		SetBackendError("cuModuleGetFunction(KernelGen) failed: %s", errbuf);
		ReleaseBackend();
		return false;
	}

	cu_err = cuFuncSetAttribute(SassKernelA, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)Kparams.KernelA_LDS_Size);
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		SetBackendError("cuFuncSetAttribute(KernelA) failed: %s", errbuf);
		ReleaseBackend();
		return false;
	}
	cu_err = cuFuncSetAttribute(SassKernelB, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)Kparams.KernelB_LDS_Size);
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		SetBackendError("cuFuncSetAttribute(KernelB) failed: %s", errbuf);
		ReleaseBackend();
		return false;
	}
	cu_err = cuFuncSetAttribute(SassKernelC, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)Kparams.KernelC_LDS_Size);
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		SetBackendError("cuFuncSetAttribute(KernelC) failed: %s", errbuf);
		ReleaseBackend();
		return false;
	}

	// Favor shared memory when hardware supports carveout tuning.
	const int shared_carveout = 100;
	CUresult carve_rc = cuFuncSetAttribute(SassKernelA, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, shared_carveout);
	if (carve_rc != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(carve_rc, errbuf, sizeof(errbuf));
		printf("GPU %d: warning: KernelA shared carveout hint failed: %s\r\n", CudaIndex, errbuf);
	}
	carve_rc = cuFuncSetAttribute(SassKernelB, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, shared_carveout);
	if (carve_rc != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(carve_rc, errbuf, sizeof(errbuf));
		printf("GPU %d: warning: KernelB shared carveout hint failed: %s\r\n", CudaIndex, errbuf);
	}
	carve_rc = cuFuncSetAttribute(SassKernelC, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, shared_carveout);
	if (carve_rc != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(carve_rc, errbuf, sizeof(errbuf));
		printf("GPU %d: warning: KernelC shared carveout hint failed: %s\r\n", CudaIndex, errbuf);
	}

	CUdeviceptr jmp2_sym = 0;
	size_t jmp2_sym_size = 0;
	const char* global_names[] = { "jmp2_table", "_ZL10jmp2_table" };
	bool global_found = false;
	for (int i = 0; i < 2; i++)
	{
		cu_err = cuModuleGetGlobal(&jmp2_sym, &jmp2_sym_size, SassModule, global_names[i]);
		if (cu_err == CUDA_SUCCESS)
		{
			global_found = true;
			break;
		}
	}
	if (!global_found)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		SetBackendError("cuModuleGetGlobal(jmp2_table) failed: %s", errbuf);
		ReleaseBackend();
		return false;
	}

	if (jmp2_sym_size < (size_t)(JMP_CNT * 64))
	{
		SetBackendError("jmp2_table symbol too small (%llu)", (unsigned long long)jmp2_sym_size);
		ReleaseBackend();
		return false;
	}

	cu_err = cuMemcpyHtoD(jmp2_sym, jmp2_table, JMP_CNT * 64);
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		SetBackendError("cuMemcpyHtoD(jmp2_table) failed: %s", errbuf);
		ReleaseBackend();
		return false;
	}

	strncpy(LoadedSassPath, cubin_path, sizeof(LoadedSassPath) - 1);
	LoadedSassPath[sizeof(LoadedSassPath) - 1] = 0;
	return true;
}

void RCGpuKang::ReleaseBackend()
{
	if (SassModule)
	{
		cuModuleUnload(SassModule);
		SassModule = nullptr;
	}
	SassKernelA = nullptr;
	SassKernelB = nullptr;
	SassKernelC = nullptr;
	SassKernelGen = nullptr;
}

cudaError_t RCGpuKang::LaunchKernelGen()
{
	void* args[] = { &Kparams };
	CUresult cu_err = cuLaunchKernel(
		SassKernelGen,
		Kparams.BlockCnt, 1, 1,
		Kparams.BlockSize, 1, 1,
		0,
		(CUstream)ExecStream,
		args,
		0);
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		printf("GPU %d, SASS KernelGen launch failed: %s\r\n", CudaIndex, errbuf);
		return cudaErrorUnknown;
	}
	return cudaSuccess;
}

cudaError_t RCGpuKang::LaunchKernelABC()
{
	void* args[] = { &Kparams };
	CUresult cu_err = cuLaunchKernel(
		SassKernelA,
		Kparams.BlockCnt, 1, 1,
		Kparams.BlockSize, 1, 1,
		Kparams.KernelA_LDS_Size,
		(CUstream)ExecStream,
		args,
		0);
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		printf("GPU %d, SASS KernelA launch failed: %s\r\n", CudaIndex, errbuf);
		return cudaErrorUnknown;
	}

	cu_err = cuLaunchKernel(
		SassKernelB,
		Kparams.BlockCnt, 1, 1,
		Kparams.BlockSize, 1, 1,
		Kparams.KernelB_LDS_Size,
		(CUstream)ExecStream,
		args,
		0);
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		printf("GPU %d, SASS KernelB launch failed: %s\r\n", CudaIndex, errbuf);
		return cudaErrorUnknown;
	}

	cu_err = cuLaunchKernel(
		SassKernelC,
		Kparams.BlockCnt, 1, 1,
		Kparams.BlockSize, 1, 1,
		Kparams.KernelC_LDS_Size,
		(CUstream)ExecStream,
		args,
		0);
	if (cu_err != CUDA_SUCCESS)
	{
		char errbuf[256];
		FormatCuError(cu_err, errbuf, sizeof(errbuf));
		printf("GPU %d, SASS KernelC launch failed: %s\r\n", CudaIndex, errbuf);
		return cudaErrorUnknown;
	}

	return cudaSuccess;
}

//executes in main thread
bool RCGpuKang::Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3)
{
	PntToSolve = _PntToSolve;
	Range = _Range;
	DP = _DP;
	EcJumps1 = _EcJumps1;
	EcJumps2 = _EcJumps2;
	EcJumps3 = _EcJumps3;
	StopFlag = false;
	Failed = false;
	u64 total_mem = 0;
	memset(&Kparams, 0, sizeof(Kparams));
	memset(dbg, 0, sizeof(dbg));
	memset(SpeedStats, 0, sizeof(SpeedStats));
	cur_stats_ind = 0;
	RndPnts = nullptr;
	DPs_out = nullptr;
	ExecStream = nullptr;
	DpCountHost = 0;
	DpBufferPinned = false;
	SassModule = nullptr;
	SassKernelA = nullptr;
	SassKernelB = nullptr;
	SassKernelC = nullptr;
	SassKernelGen = nullptr;
	BackendError[0] = 0;
	LoadedSassPath[0] = 0;
	LaunchConfigError[0] = 0;
	LaunchConfigSource[0] = 0;
	LaunchContractBlockSize = 0;
	LaunchContractGroupCnt = 0;
	LaunchGroupLocked = false;
	LaunchStrictMode = true;

	auto FailPrepare = [&]() -> bool
	{
		Release();
		return false;
	};

	cudaError_t err;
	err = cudaSetDevice(CudaIndex);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaSetDevice failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}
	err = cudaStreamCreateWithFlags(&ExecStream, cudaStreamNonBlocking);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaStreamCreateWithFlags failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	ResolveLaunchConfig(Kparams.BlockCnt, Kparams.BlockSize, Kparams.GroupCnt);
	if (LaunchConfigError[0])
	{
		printf("GPU %d, launch profile error: %s\r\n", CudaIndex, LaunchConfigError);
		return FailPrepare();
	}
	if (gDpExportMode == DP_EXPORT_WILD)
		Kparams.RunMode = KANG_MODE_EXPORT_WILD;
	else if (gDpExportMode == DP_EXPORT_TAME)
		Kparams.RunMode = KANG_MODE_EXPORT_TAME;
	else if (gDpExportMode == DP_EXPORT_BOTH)
		Kparams.RunMode = KANG_MODE_EXPORT_BOTH;
	else if (gGenMode)
		Kparams.RunMode = KANG_MODE_GEN_TAME;
	else
		Kparams.RunMode = KANG_MODE_MAIN;
	if ((Kparams.RunMode == KANG_MODE_MAIN) && !IsEnvSet("RCK_GROUP_CNT") && !LaunchGroupLocked)
		TuneGroupCntForMainSolve(Kparams.GroupCnt, Kparams.BlockCnt, Kparams.BlockSize, Range, DP, CudaIndex);

	KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
	Kparams.KangCnt = KangCnt;
	Kparams.DP = DP;
	Kparams.KernelA_LDS_Size = 64 * JMP_CNT + 16 * Kparams.BlockSize;
	Kparams.KernelB_LDS_Size = 64 * JMP_CNT;
	Kparams.KernelC_LDS_Size = 96 * JMP_CNT;

//allocate gpu mem
	u64 size;
	if (!IsOldGpu)
	{
		//L2	
		int L2size = Kparams.KangCnt * (3 * 32);
		total_mem += L2size;
			err = cudaMalloc((void**)&Kparams.L2, L2size);
			if (err != cudaSuccess)
			{
				printf("GPU %d, Allocate L2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
				return FailPrepare();
			}
		size = L2size;
		if (size > persistingL2CacheMaxSize)
			size = persistingL2CacheMaxSize;
			err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); // set max allowed size for L2
			if (err != cudaSuccess)
			{
				printf("GPU %d, cudaDeviceSetLimit failed: %s\n", CudaIndex, cudaGetErrorString(err));
				return FailPrepare();
			}
		//persisting for L2
		cudaStreamAttrValue stream_attribute;                                                   
		stream_attribute.accessPolicyWindow.base_ptr = Kparams.L2;
		stream_attribute.accessPolicyWindow.num_bytes = size;										
		stream_attribute.accessPolicyWindow.hitRatio = 1.0;                                     
		stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;             
		stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  	
			err = cudaStreamSetAttribute(ExecStream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
			if (err != cudaSuccess)
			{
				printf("GPU %d, cudaStreamSetAttribute failed: %s\n", CudaIndex, cudaGetErrorString(err));
				return FailPrepare();
			}
		}
	size = MAX_DP_CNT * GPU_DP_SIZE + 16;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.DPs_out, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate GpuOut memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	size = KangCnt * 96;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.Kangs, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate pKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps1, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps2, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps3 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	size = 2 * (u64)KangCnt * STEP_CNT;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.JumpsList, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate JumpsList memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32)); //we store 16bytes of X
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.DPTable, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate DPTable memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	size = mpCnt * Kparams.BlockSize * sizeof(u64);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.L1S2, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate L1S2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	size = (u64)KangCnt * MD_LEN * (2 * 32);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LastPnts, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	size = (u64)KangCnt * MD_LEN * sizeof(u64);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LoopTable, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LoopTable memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	total_mem += 1024;
	err = cudaMalloc((void**)&Kparams.dbg_buf, 1024);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate dbg_buf memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	size = sizeof(u32) * KangCnt + 8;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LoopedKangs, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LoopedKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	size_t host_out_bytes = MAX_DP_CNT * GPU_DP_SIZE;
	err = cudaMallocHost((void**)&DPs_out, host_out_bytes);
	if (err == cudaSuccess)
	{
		DpBufferPinned = true;
	}
	else
	{
		DPs_out = (u32*)malloc(host_out_bytes);
		DpBufferPinned = false;
		if (!DPs_out)
		{
			printf("GPU %d, host DP buffer allocation failed\n", CudaIndex);
			return FailPrepare();
		}
	}

//jmp1
	std::vector<u64> jmp_buf(JMP_CNT * 12);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(jmp_buf.data() + i * 12, EcJumps1[i].p.x.data, 32);
		memcpy(jmp_buf.data() + i * 12 + 4, EcJumps1[i].p.y.data, 32);
		memcpy(jmp_buf.data() + i * 12 + 8, EcJumps1[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps1, jmp_buf.data(), JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps1 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}
//jmp2
	std::vector<u64> jmp2_buf(JMP_CNT * 12);
	std::vector<u64> jmp2_table(JMP_CNT * 8);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(jmp2_buf.data() + i * 12, EcJumps2[i].p.x.data, 32);
		memcpy(jmp2_table.data() + i * 8, EcJumps2[i].p.x.data, 32);
		memcpy(jmp2_buf.data() + i * 12 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(jmp2_table.data() + i * 8 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(jmp2_buf.data() + i * 12 + 8, EcJumps2[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps2, jmp2_buf.data(), JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps2 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	if (!InitSassBackend(jmp2_table.data()))
	{
		printf("GPU %d, SASS backend init failed: %s\r\n", CudaIndex, BackendError[0] ? BackendError : "unknown");
		return FailPrepare();
	}
//jmp3
	std::vector<u64> jmp3_buf(JMP_CNT * 12);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(jmp3_buf.data() + i * 12, EcJumps3[i].p.x.data, 32);
		memcpy(jmp3_buf.data() + i * 12 + 4, EcJumps3[i].p.y.data, 32);
		memcpy(jmp3_buf.data() + i * 12 + 8, EcJumps3[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps3, jmp3_buf.data(), JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps3 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return FailPrepare();
	}

	printf("GPU %d: backend=sass, cubin=%s\r\n", CudaIndex, LoadedSassPath);
	printf("GPU %d: launch profile blocks=%u block_size=%u groups=%u (SM %d.%d)\r\n",
		CudaIndex,
		Kparams.BlockCnt,
		Kparams.BlockSize,
		Kparams.GroupCnt,
		SmMajor,
		SmMinor);
	printf(
		"GPU %d: launch contract block=%u group=%u source=%s strict=%s\r\n",
		CudaIndex,
		LaunchContractBlockSize,
		LaunchContractGroupCnt,
		LaunchConfigSource[0] ? LaunchConfigSource : "unknown",
		LaunchStrictMode ? "on" : "off");

	printf("GPU %d: allocated %llu MB, %d kangaroos. OldGpuMode: %s\r\n", CudaIndex, total_mem / (1024 * 1024), KangCnt, IsOldGpu ? "Yes" : "No");
	return true;
}

void RCGpuKang::Release()
{
	if (ExecStream)
		cudaStreamSynchronize(ExecStream);
	ReleaseBackend();
	if (RndPnts)
	{
		free(RndPnts);
		RndPnts = nullptr;
	}
	if (DPs_out)
	{
		if (DpBufferPinned)
			cudaFreeHost(DPs_out);
		else
			free(DPs_out);
		DPs_out = nullptr;
	}
	cudaFree(Kparams.LoopedKangs);
	cudaFree(Kparams.dbg_buf);
	cudaFree(Kparams.LoopTable);
	cudaFree(Kparams.LastPnts);
	cudaFree(Kparams.L1S2);
	cudaFree(Kparams.DPTable);
	cudaFree(Kparams.JumpsList);
	cudaFree(Kparams.Jumps3);
	cudaFree(Kparams.Jumps2);
	cudaFree(Kparams.Jumps1);
	cudaFree(Kparams.Kangs);
	cudaFree(Kparams.DPs_out);
	if (!IsOldGpu)
		cudaFree(Kparams.L2);
	if (ExecStream)
	{
		cudaStreamDestroy(ExecStream);
		ExecStream = nullptr;
	}
	DpBufferPinned = false;
	DpCountHost = 0;
	memset(&Kparams, 0, sizeof(Kparams));
}

void RCGpuKang::Stop()
{
	StopFlag = true;
}

void RCGpuKang::GenerateRndDistances()
{
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		if ((Kparams.RunMode == KANG_MODE_GEN_TAME) || (Kparams.RunMode == KANG_MODE_EXPORT_TAME))
			d.RndBits(Range - 4); //TAME kangs
		else if ((Kparams.RunMode == KANG_MODE_MAIN) || (Kparams.RunMode == KANG_MODE_EXPORT_BOTH))
		{
			if (i < KangCnt / 3)
				d.RndBits(Range - 4); //TAME kangs
			else
			{
				d.RndBits(Range - 1);
				d.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
			}
		}
		else
		{
			d.RndBits(Range - 1);
			d.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		}
		memcpy(RndPnts[i].priv, d.data, 24);
	}
}

bool RCGpuKang::Start()
{
	if (Failed)
		return false;

	cudaError_t err;
	err = cudaSetDevice(CudaIndex);
	if (err != cudaSuccess)
		return false;

	HalfRange.Set(1);
	HalfRange.ShiftLeft(Range - 1);
	PntHalfRange = ec.MultiplyG(HalfRange);
	NegPntHalfRange = PntHalfRange;
	NegPntHalfRange.y.NegModP();

	PntA = ec.AddPoints(PntToSolve, NegPntHalfRange);
	PntB = PntA;
	PntB.y.NegModP();

	RndPnts = (TPointPriv*)malloc(KangCnt * 96);
	if (!RndPnts)
	{
		printf("GPU %d, host kangaroo buffer allocation failed\n", CudaIndex);
		return false;
	}
	GenerateRndDistances();
/* 
	//we can calc start points on CPU
	for (int i = 0; i < KangCnt; i++)
	{
		EcInt d;
		memcpy(d.data, RndPnts[i].priv, 24);
		d.data[3] = 0;
		d.data[4] = 0;
		EcPoint p = ec.MultiplyG(d);
		memcpy(RndPnts[i].x, p.x.data, 32);
		memcpy(RndPnts[i].y, p.y.data, 32);
	}
	for (int i = KangCnt / 3; i < 2 * KangCnt / 3; i++)
	{
		EcPoint p;
		p.LoadFromBuffer64((u8*)RndPnts[i].x);
		p = ec.AddPoints(p, PntA);
		p.SaveToBuffer64((u8*)RndPnts[i].x);
	}
	for (int i = 2 * KangCnt / 3; i < KangCnt; i++)
	{
		EcPoint p;
		p.LoadFromBuffer64((u8*)RndPnts[i].x);
		p = ec.AddPoints(p, PntB);
		p.SaveToBuffer64((u8*)RndPnts[i].x);
	}
	//copy to gpu
	err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
/**/
	//but it's faster to calc them on GPU
	u8 buf_PntA[64], buf_PntB[64];
	PntA.SaveToBuffer64(buf_PntA);
	PntB.SaveToBuffer64(buf_PntB);
	for (int i = 0; i < KangCnt; i++)
	{
		if (Kparams.RunMode == KANG_MODE_EXPORT_WILD)
		{
			if (i < KangCnt / 2)
				memcpy(RndPnts[i].x, buf_PntA, 64);
			else
				memcpy(RndPnts[i].x, buf_PntB, 64);
		}
		else if ((Kparams.RunMode == KANG_MODE_GEN_TAME) || (Kparams.RunMode == KANG_MODE_EXPORT_TAME))
		{
			memset(RndPnts[i].x, 0, 64);
		}
		else
		{
			if (i < KangCnt / 3)
				memset(RndPnts[i].x, 0, 64);
			else if (i < 2 * KangCnt / 3)
				memcpy(RndPnts[i].x, buf_PntA, 64);
			else
				memcpy(RndPnts[i].x, buf_PntB, 64);
		}
	}
	//copy to gpu
	err = cudaMemcpyAsync(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice, ExecStream);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	err = LaunchKernelGen();
	if (err != cudaSuccess)
	{
		printf("GPU %d, LaunchKernelGen failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	err = cudaMemsetAsync(Kparams.L1S2, 0, mpCnt * Kparams.BlockSize * 8, ExecStream);
	if (err != cudaSuccess)
		return false;
	cudaMemsetAsync(Kparams.dbg_buf, 0, 1024, ExecStream);
	cudaMemsetAsync(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64), ExecStream);
	err = cudaStreamSynchronize(ExecStream);
	if (err != cudaSuccess)
		return false;
	return true;
}

#ifdef DEBUG_MODE
int RCGpuKang::Dbg_CheckKangs()
{
	int kang_size = mpCnt * Kparams.BlockSize * Kparams.GroupCnt * 96;
	u64* kangs = (u64*)malloc(kang_size);
	cudaError_t err = cudaMemcpy(kangs, Kparams.Kangs, kang_size, cudaMemcpyDeviceToHost);
	int res = 0;
	for (int i = 0; i < KangCnt; i++)
	{
		EcPoint Pnt, p;
		Pnt.LoadFromBuffer64((u8*)&kangs[i * 12 + 0]);
		EcInt dist;
		dist.Set(0);
		memcpy(dist.data, &kangs[i * 12 + 8], 24);
		bool neg = false;
		if (dist.data[2] >> 63)
		{
			neg = true;
			memset(((u8*)dist.data) + 24, 0xFF, 16);
			dist.Neg();
		}
		p = ec.MultiplyG_Fast(dist);
		if (neg)
			p.y.NegModP();
		if (i < KangCnt / 3)
			p = p;
		else
			if (i < 2 * KangCnt / 3)
				p = ec.AddPoints(PntA, p);
			else
				p = ec.AddPoints(PntB, p);
		if (!p.IsEqual(Pnt))
			res++;
	}
	free(kangs);
	return res;
}
#endif

extern std::atomic<u32> gTotalErrors;

//executes in separate thread
void RCGpuKang::Execute()
{
	cudaSetDevice(CudaIndex);

	if (!Start())
	{
		Release();
		gTotalErrors.fetch_add(1, std::memory_order_relaxed);
		return;
	}
#ifdef DEBUG_MODE
	u64 iter = 1;
#endif
	cudaError_t err;	
	while (!StopFlag)
	{
		u64 t1 = GetTickCount64();
		err = cudaMemsetAsync(Kparams.DPs_out, 0, 4, ExecStream);
		if (err != cudaSuccess)
		{
			gTotalErrors.fetch_add(1, std::memory_order_relaxed);
			break;
		}
		err = cudaMemsetAsync(Kparams.DPTable, 0, KangCnt * sizeof(u32), ExecStream);
		if (err != cudaSuccess)
		{
			gTotalErrors.fetch_add(1, std::memory_order_relaxed);
			break;
		}
		err = cudaMemsetAsync(Kparams.LoopedKangs, 0, 8, ExecStream);
		if (err != cudaSuccess)
		{
			gTotalErrors.fetch_add(1, std::memory_order_relaxed);
			break;
		}
		err = LaunchKernelABC();
		if (err != cudaSuccess)
		{
			printf("GPU %d, LaunchKernelABC failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
			gTotalErrors.fetch_add(1, std::memory_order_relaxed);
			break;
		}
		err = cudaMemcpyAsync(&DpCountHost, Kparams.DPs_out, 4, cudaMemcpyDeviceToHost, ExecStream);
		if (err != cudaSuccess)
		{
			printf("GPU %d, cudaMemcpyAsync(dp_count) failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
			gTotalErrors.fetch_add(1, std::memory_order_relaxed);
			break;
		}
		err = cudaStreamSynchronize(ExecStream);
		if (err != cudaSuccess)
		{
			printf("GPU %d, cudaStreamSynchronize failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
			gTotalErrors.fetch_add(1, std::memory_order_relaxed);
			break;
		}
		int cnt = (int)DpCountHost;
		
		if (cnt >= MAX_DP_CNT)
		{
			cnt = MAX_DP_CNT;
			printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", CudaIndex);
		}
		u64 pnt_cnt = (u64)KangCnt * STEP_CNT;

		if (cnt)
		{
			err = cudaMemcpyAsync(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, cudaMemcpyDeviceToHost, ExecStream);
			if (err != cudaSuccess)
			{
				gTotalErrors.fetch_add(1, std::memory_order_relaxed);
				break;
			}
			err = cudaStreamSynchronize(ExecStream);
			if (err != cudaSuccess)
			{
				gTotalErrors.fetch_add(1, std::memory_order_relaxed);
				break;
			}
			AddPointsToList(DPs_out, cnt, (u64)KangCnt * STEP_CNT);
		}

#ifdef DEBUG_MODE
		//dbg
		cudaMemcpyAsync(dbg, Kparams.dbg_buf, 1024, cudaMemcpyDeviceToHost, ExecStream);
		u32 lcnt = 0;
		cudaMemcpyAsync(&lcnt, Kparams.LoopedKangs, 4, cudaMemcpyDeviceToHost, ExecStream);
		cudaStreamSynchronize(ExecStream);
		//printf("GPU %d, Looped: %d\r\n", CudaIndex, lcnt);
#endif

		u64 t2 = GetTickCount64();
		u64 tm = t2 - t1;
		if (!tm)
			tm = 1;
		int cur_speed = (int)(pnt_cnt / (tm * 1000));
		//printf("GPU %d kernel time %d ms, speed %d MH\r\n", CudaIndex, (int)tm, cur_speed);

		SpeedStats[cur_stats_ind] = cur_speed;
		cur_stats_ind = (cur_stats_ind + 1) % STATS_WND_SIZE;

#ifdef DEBUG_MODE
		if ((iter % 300) == 0)
		{
			int corr_cnt = Dbg_CheckKangs();
			if (corr_cnt)
			{
				printf("DBG: GPU %d, KANGS CORRUPTED: %d\r\n", CudaIndex, corr_cnt);
				gTotalErrors.fetch_add(1, std::memory_order_relaxed);
			}
			else
				printf("DBG: GPU %d, ALL KANGS OK!\r\n", CudaIndex);
		}
		iter++;
#endif
	}

	Release();
}

int RCGpuKang::GetStatsSpeed()
{
	int res = SpeedStats[0];
	for (int i = 1; i < STATS_WND_SIZE; i++)
		res += SpeedStats[i];
	return res / STATS_WND_SIZE;
}
