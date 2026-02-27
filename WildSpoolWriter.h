// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "defs.h"

class WildSpoolWriter
{
public:
	WildSpoolWriter();
	~WildSpoolWriter();

	bool Start(
		const std::string& spoolDir,
		const std::string& workerId,
		const std::string& sessionTag,
		const std::string& targetPubkey,
		const std::string& startOffset,
		u32 rangeBits,
		u32 dpBits,
		u32 flushRecords,
		u32 flushSec);
	void Enqueue(const u8* xPrefix, const u8* distanceRaw24, u8 dpType);
	void StopAndFlush();

	u64 GetWrittenRecords() const;
	u64 GetWrittenFiles() const;
	u64 GetDroppedRecords() const;
	u64 GetPendingRecords() const;

private:
	struct RawRecord
	{
		u8 xPrefix[12];
		u8 distanceRaw24[24];
		u8 dpType;
		u8 flags;
		u16 reserved;
	};

	std::string spoolDir;
	std::string workerId;
	std::string sessionTag;
	std::string targetPubkey;
	std::string startOffset;
	u32 rangeBits;
	u32 dpBits;
	u32 flushRecords;
	u32 flushSec;
	size_t maxQueueRecords;
	u64 chunkSeq;

	std::deque<RawRecord> queue;
	std::vector<RawRecord> chunk;
	mutable std::mutex mtx;
	std::condition_variable cvData;
	std::condition_variable cvSpace;
	std::thread writerThread;
	bool stopRequested;
	bool isRunning;

	std::atomic<u64> writtenRecords;
	std::atomic<u64> writtenFiles;
	std::atomic<u64> droppedRecords;

	bool EnsureSpoolDir();
	void WriterLoop();
	bool WriteChunk(const std::vector<RawRecord>& records);
};
