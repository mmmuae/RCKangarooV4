// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"

#include "GpuKang.h"

cudaError_t cuSetGpuParams(TKparams Kparams, u64* _jmp2_table);
void CallGpuKernelGen(TKparams Kparams);
void CallGpuKernelABC(TKparams Kparams);
void AddPointsToList(u32* data, int cnt, u64 ops_cnt);
extern bool gGenMode; //tames generation mode
extern u32 gDpExportMode;
extern u32 gKernelBackendMode;

static const char* BackendModeName(u32 mode)
{
	switch (mode)
	{
	case GPU_BACKEND_SASS:
		return "sass";
	case GPU_BACKEND_CUDA:
		return "cuda";
	default:
		return "auto";
	}
}

static bool FileExists(const char* path)
{
	FILE* fp = fopen(path, "rb");
	if (!fp)
		return false;
	fclose(fp);
	return true;
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

	const char* env_dir = getenv("RCK_SASS_DIR");
	if (env_dir && env_dir[0])
	{
		snprintf(out_path, out_size, "%s/%s/rckangaroo_kernels.cubin", env_dir, arch_tag);
		if (FileExists(out_path))
			return true;
	}

	snprintf(out_path, out_size, "sass/%s/rckangaroo_kernels.cubin", arch_tag);
	return FileExists(out_path);
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

int RCGpuKang::CalcKangCnt()
{
	Kparams.BlockCnt = mpCnt;
	Kparams.BlockSize = IsOldGpu ? 512 : 256;
	Kparams.GroupCnt = IsOldGpu ? 64 : 24;
	return Kparams.BlockSize* Kparams.GroupCnt* Kparams.BlockCnt;
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
		SetBackendError("no cubin for sm%d%d in ./sass or RCK_SASS_DIR", dev_prop.major, dev_prop.minor);
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

bool RCGpuKang::InitBackend(const u64* jmp2_table)
{
	ActiveBackend = GPU_BACKEND_CUDA;
	SetBackendError(nullptr);
	LoadedSassPath[0] = 0;

	if (gKernelBackendMode == GPU_BACKEND_CUDA)
		return true;

	bool sass_ready = InitSassBackend(jmp2_table);
	if (sass_ready)
	{
		ActiveBackend = GPU_BACKEND_SASS;
		return true;
	}

	if (gKernelBackendMode == GPU_BACKEND_SASS)
		return false;

	// auto mode fallback path
	printf("GPU %d: SASS backend unavailable (%s), fallback to CUDA runtime kernels.\r\n", CudaIndex, BackendError[0] ? BackendError : "unknown");
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
	if (ActiveBackend == GPU_BACKEND_SASS)
	{
		void* args[] = { &Kparams };
		CUresult cu_err = cuLaunchKernel(
			SassKernelGen,
			Kparams.BlockCnt, 1, 1,
			Kparams.BlockSize, 1, 1,
			0,
			0,
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

	CallGpuKernelGen(Kparams);
	return cudaGetLastError();
}

cudaError_t RCGpuKang::LaunchKernelABC()
{
	if (ActiveBackend == GPU_BACKEND_SASS)
	{
		void* args[] = { &Kparams };
		CUresult cu_err = cuLaunchKernel(
			SassKernelA,
			Kparams.BlockCnt, 1, 1,
			Kparams.BlockSize, 1, 1,
			Kparams.KernelA_LDS_Size,
			0,
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
			0,
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
			0,
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

	CallGpuKernelABC(Kparams);
	return cudaGetLastError();
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
	memset(dbg, 0, sizeof(dbg));
	memset(SpeedStats, 0, sizeof(SpeedStats));
	cur_stats_ind = 0;
	ActiveBackend = GPU_BACKEND_CUDA;
	SassModule = nullptr;
	SassKernelA = nullptr;
	SassKernelB = nullptr;
	SassKernelC = nullptr;
	SassKernelGen = nullptr;
	BackendError[0] = 0;
	LoadedSassPath[0] = 0;

	cudaError_t err;
	err = cudaSetDevice(CudaIndex);
	if (err != cudaSuccess)
		return false;

	Kparams.BlockCnt = mpCnt;
	Kparams.BlockSize = IsOldGpu ? BLOCK_SIZE_OLD_GPU : BLOCK_SIZE_NEW_GPU;
	Kparams.GroupCnt = IsOldGpu ? PNT_GROUP_OLD_GPU : PNT_GROUP_NEW_GPU;
	KangCnt = Kparams.BlockSize * Kparams.GroupCnt * Kparams.BlockCnt;
	Kparams.KangCnt = KangCnt;
	Kparams.DP = DP;
	Kparams.KernelA_LDS_Size = 64 * JMP_CNT + 16 * Kparams.BlockSize;
	Kparams.KernelB_LDS_Size = 64 * JMP_CNT;
	Kparams.KernelC_LDS_Size = 96 * JMP_CNT;
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
			return false;
		}
		size = L2size;
		if (size > persistingL2CacheMaxSize)
			size = persistingL2CacheMaxSize;
		err = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size); // set max allowed size for L2
		//persisting for L2
		cudaStreamAttrValue stream_attribute;                                                   
		stream_attribute.accessPolicyWindow.base_ptr = Kparams.L2;
		stream_attribute.accessPolicyWindow.num_bytes = size;										
		stream_attribute.accessPolicyWindow.hitRatio = 1.0;                                     
		stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;             
		stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;  	
		err = cudaStreamSetAttribute(NULL, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
		if (err != cudaSuccess)
		{
			printf("GPU %d, cudaStreamSetAttribute failed: %s\n", CudaIndex, cudaGetErrorString(err));
			return false;
		}
	}
	size = MAX_DP_CNT * GPU_DP_SIZE + 16;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.DPs_out, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate GpuOut memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = KangCnt * 96;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.Kangs, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate pKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps1, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps2, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps1 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += JMP_CNT * 96;
	err = cudaMalloc((void**)&Kparams.Jumps3, JMP_CNT * 96);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate Jumps3 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = 2 * (u64)KangCnt * STEP_CNT;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.JumpsList, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate JumpsList memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * (16 * DPTABLE_MAX_CNT + sizeof(u32)); //we store 16bytes of X
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.DPTable, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate DPTable memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = mpCnt * Kparams.BlockSize * sizeof(u64);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.L1S2, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate L1S2 memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * (2 * 32);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LastPnts, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = (u64)KangCnt * MD_LEN * sizeof(u64);
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LoopTable, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LastPnts memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	total_mem += 1024;
	err = cudaMalloc((void**)&Kparams.dbg_buf, 1024);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate dbg_buf memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	size = sizeof(u32) * KangCnt + 8;
	total_mem += size;
	err = cudaMalloc((void**)&Kparams.LoopedKangs, size);
	if (err != cudaSuccess)
	{
		printf("GPU %d Allocate LoopedKangs memory failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}

	DPs_out = (u32*)malloc(MAX_DP_CNT * GPU_DP_SIZE);

//jmp1
	u64* buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps1[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps1[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps1[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps1, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps1 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);
//jmp2
	buf = (u64*)malloc(JMP_CNT * 96);
	u64* jmp2_table = (u64*)malloc(JMP_CNT * 64);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps2[i].p.x.data, 32);
		memcpy(jmp2_table + i * 8, EcJumps2[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(jmp2_table + i * 8 + 4, EcJumps2[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps2[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps2, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps2 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);

	if (!InitBackend(jmp2_table))
	{
		free(jmp2_table);
		printf("GPU %d, backend init failed (requested: %s): %s\r\n", CudaIndex, BackendModeName(gKernelBackendMode), BackendError[0] ? BackendError : "unknown");
		return false;
	}

	if (ActiveBackend == GPU_BACKEND_CUDA)
	{
		err = cuSetGpuParams(Kparams, jmp2_table);
		if (err != cudaSuccess)
		{
			free(jmp2_table);
			printf("GPU %d, cuSetGpuParams failed: %s!\r\n", CudaIndex, cudaGetErrorString(err));
			return false;
		}
	}
	free(jmp2_table);
//jmp3
	buf = (u64*)malloc(JMP_CNT * 96);
	for (int i = 0; i < JMP_CNT; i++)
	{
		memcpy(buf + i * 12, EcJumps3[i].p.x.data, 32);
		memcpy(buf + i * 12 + 4, EcJumps3[i].p.y.data, 32);
		memcpy(buf + i * 12 + 8, EcJumps3[i].dist.data, 32);
	}
	err = cudaMemcpy(Kparams.Jumps3, buf, JMP_CNT * 96, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("GPU %d, cudaMemcpy Jumps3 failed: %s\n", CudaIndex, cudaGetErrorString(err));
		return false;
	}
	free(buf);

	if (ActiveBackend == GPU_BACKEND_SASS)
		printf("GPU %d: backend=sass, cubin=%s\r\n", CudaIndex, LoadedSassPath);
	else
		printf("GPU %d: backend=cuda\r\n", CudaIndex);

	printf("GPU %d: allocated %llu MB, %d kangaroos. OldGpuMode: %s\r\n", CudaIndex, total_mem / (1024 * 1024), KangCnt, IsOldGpu ? "Yes" : "No");
	return true;
}

void RCGpuKang::Release()
{
	ReleaseBackend();
	free(RndPnts);
	free(DPs_out);
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
	err = cudaMemcpy(Kparams.Kangs, RndPnts, KangCnt * 96, cudaMemcpyHostToDevice);
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

	err = cudaMemset(Kparams.L1S2, 0, mpCnt * Kparams.BlockSize * 8);
	if (err != cudaSuccess)
		return false;
	cudaMemset(Kparams.dbg_buf, 0, 1024);
	cudaMemset(Kparams.LoopTable, 0, KangCnt * MD_LEN * sizeof(u64));
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

extern u32 gTotalErrors;

//executes in separate thread
void RCGpuKang::Execute()
{
	cudaSetDevice(CudaIndex);

	if (!Start())
	{
		ReleaseBackend();
		gTotalErrors++;
		return;
	}
#ifdef DEBUG_MODE
	u64 iter = 1;
#endif
	cudaError_t err;	
	while (!StopFlag)
	{
		u64 t1 = GetTickCount64();
		cudaMemset(Kparams.DPs_out, 0, 4);
		cudaMemset(Kparams.DPTable, 0, KangCnt * sizeof(u32));
		cudaMemset(Kparams.LoopedKangs, 0, 8);
		err = LaunchKernelABC();
		if (err != cudaSuccess)
		{
			printf("GPU %d, LaunchKernelABC failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
			gTotalErrors++;
			break;
		}
		int cnt;
		err = cudaMemcpy(&cnt, Kparams.DPs_out, 4, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			printf("GPU %d, CallGpuKernel failed: %s\r\n", CudaIndex, cudaGetErrorString(err));
			gTotalErrors++;
			break;
		}
		
		if (cnt >= MAX_DP_CNT)
		{
			cnt = MAX_DP_CNT;
			printf("GPU %d, gpu DP buffer overflow, some points lost, increase DP value!\r\n", CudaIndex);
		}
		u64 pnt_cnt = (u64)KangCnt * STEP_CNT;

		if (cnt)
		{
			err = cudaMemcpy(DPs_out, Kparams.DPs_out + 4, cnt * GPU_DP_SIZE, cudaMemcpyDeviceToHost);
			if (err != cudaSuccess)
			{
				gTotalErrors++;
				break;
			}
			AddPointsToList(DPs_out, cnt, (u64)KangCnt * STEP_CNT);
		}

		//dbg
		cudaMemcpy(dbg, Kparams.dbg_buf, 1024, cudaMemcpyDeviceToHost);

		u32 lcnt;
		cudaMemcpy(&lcnt, Kparams.LoopedKangs, 4, cudaMemcpyDeviceToHost);
		//printf("GPU %d, Looped: %d\r\n", CudaIndex, lcnt);

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
				gTotalErrors++;
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
