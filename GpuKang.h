// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#pragma once

#include "cuda_runtime.h"
#include "cuda.h"
#include "Ec.h"

#define STATS_WND_SIZE	16

struct EcJMP
{
	EcPoint p;
	EcInt dist;
};

//96bytes size
struct TPointPriv
{
	u64 x[4];
	u64 y[4];
	u64 priv[4];
};

class RCGpuKang
{
private:
	bool StopFlag;
	EcPoint PntToSolve;
	int Range; //in bits
	int DP; //in bits
	Ec ec;

	u32* DPs_out;
	TKparams Kparams;

	EcInt HalfRange;
	EcPoint PntHalfRange;
	EcPoint NegPntHalfRange;
	TPointPriv* RndPnts;
	EcJMP* EcJumps1;
	EcJMP* EcJumps2;
	EcJMP* EcJumps3;

	EcPoint PntA;
	EcPoint PntB;

	int cur_stats_ind;
	int SpeedStats[STATS_WND_SIZE];

	cudaStream_t ExecStream;
	CUmodule SassModule;
	CUfunction SassKernelA;
	CUfunction SassKernelB;
	CUfunction SassKernelC;
	CUfunction SassKernelGen;
	u32 DpCountHost;
	bool DpBufferPinned;
	char BackendError[256];
	char LoadedSassPath[512];
	char LaunchConfigError[256];
	char LaunchConfigSource[96];
	u32 LaunchContractGroupCnt;
	bool LaunchGroupLocked;
	bool LaunchStrictMode;

	void GenerateRndDistances();
	void ResolveLaunchConfig(u32& blockCnt, u32& blockSize, u32& groupCnt);
	bool InitSassBackend(const u64* jmp2_table);
	void ReleaseBackend();
	cudaError_t LaunchKernelGen();
	cudaError_t LaunchKernelABC();
	void SetBackendError(const char* fmt, ...);
	bool Start();
	void Release();
#ifdef DEBUG_MODE
	int Dbg_CheckKangs();
#endif
public:
	int persistingL2CacheMaxSize;
	int CudaIndex; //gpu index in cuda
	int SmMajor;
	int SmMinor;
	int mpCnt;
	int KangCnt;
	bool Failed;
	bool IsOldGpu;

	int CalcKangCnt(int rangeBits, int dpBits, u32 runMode);
	bool Prepare(EcPoint _PntToSolve, int _Range, int _DP, EcJMP* _EcJumps1, EcJMP* _EcJumps2, EcJMP* _EcJumps3);
	void Stop();
	void Execute();

	u32 dbg[256];

	int GetStatsSpeed();
};
