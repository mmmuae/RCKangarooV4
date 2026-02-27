// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

#include "WildSpoolWriter.h"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#include "utils.h"

namespace
{
static const u32 WDP_VERSION = 1;
static const size_t WDP_WORKER_ID_LEN = 64;
static const size_t WDP_SESSION_TAG_LEN = 64;
static const size_t WDP_TARGET_PUBKEY_LEN = 132;
static const size_t WDP_START_OFFSET_LEN = 96;

#pragma pack(push, 1)
struct WdpHeader
{
	char magic[4];
	u16 version;
	u16 headerSize;
	u16 rangeBits;
	u16 dpBits;
	char workerId[WDP_WORKER_ID_LEN];
	char sessionTag[WDP_SESSION_TAG_LEN];
	char targetPubkey[WDP_TARGET_PUBKEY_LEN];
	char startOffset[WDP_START_OFFSET_LEN];
	u64 chunkSeq;
	u64 createdUnixMs;
	u32 recordCount;
	u32 bodyCrc32;
	u8 reserved[64];
};

struct WdpFooter
{
	char endMagic[8];
	u32 fileCrc32;
};
#pragma pack(pop)

u32 Crc32Update(u32 crc, const u8* data, size_t len)
{
	static u32 table[256];
	static bool ready = false;
	if (!ready)
	{
		for (u32 i = 0; i < 256; ++i)
		{
			u32 c = i;
			for (u32 j = 0; j < 8; ++j)
				c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
			table[i] = c;
		}
		ready = true;
	}
	crc = ~crc;
	for (size_t i = 0; i < len; ++i)
		crc = table[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
	return ~crc;
}

void CopyTextField(char* dst, size_t dstSize, const std::string& src)
{
	memset(dst, 0, dstSize);
	if (dstSize == 0)
		return;
	size_t copyLen = std::min(dstSize - 1, src.size());
	memcpy(dst, src.data(), copyLen);
}

bool MkdirIfNeeded(const std::string& path)
{
#ifdef _WIN32
	int rc = _mkdir(path.c_str());
#else
	int rc = mkdir(path.c_str(), 0755);
#endif
	if (rc == 0)
		return true;
	return errno == EEXIST;
}

bool EnsureDirRecursive(const std::string& path)
{
	if (path.empty())
		return false;
	std::string normalized = path;
	for (size_t i = 0; i < normalized.size(); ++i)
	{
		if (normalized[i] == '\\')
			normalized[i] = '/';
	}
	std::string current;
	if (normalized.size() >= 1 && normalized[0] == '/')
		current = "/";
#ifdef _WIN32
	if (normalized.size() >= 2 && normalized[1] == ':')
		current = normalized.substr(0, 2);
#endif
	size_t start = 0;
	while (start < normalized.size())
	{
		size_t slash = normalized.find('/', start);
		std::string part = (slash == std::string::npos) ? normalized.substr(start) : normalized.substr(start, slash - start);
		if (!part.empty())
		{
			if (!current.empty() && current[current.size() - 1] != '/')
				current.push_back('/');
			current += part;
			if (!MkdirIfNeeded(current))
				return false;
		}
		if (slash == std::string::npos)
			break;
		start = slash + 1;
	}
	return true;
}

u64 UnixTimeMs()
{
	using namespace std::chrono;
	return static_cast<u64>(duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());
}

u32 Fnv1a32(const std::string& text)
{
	u32 h = 2166136261u;
	for (size_t i = 0; i < text.size(); ++i)
	{
		h ^= static_cast<u8>(text[i]);
		h *= 16777619u;
	}
	return h;
}
} // namespace

WildSpoolWriter::WildSpoolWriter()
{
	rangeBits = 0;
	dpBits = 0;
	flushRecords = 1000000;
	flushSec = 10;
	maxQueueRecords = 0;
	chunkSeq = 1;
	stopRequested = false;
	isRunning = false;
	writtenRecords = 0;
	writtenFiles = 0;
	droppedRecords = 0;
}

WildSpoolWriter::~WildSpoolWriter()
{
	StopAndFlush();
}

bool WildSpoolWriter::Start(
	const std::string& _spoolDir,
	const std::string& _workerId,
	const std::string& _sessionTag,
	const std::string& _targetPubkey,
	const std::string& _startOffset,
	u32 _rangeBits,
	u32 _dpBits,
	u32 _flushRecords,
	u32 _flushSec)
{
	StopAndFlush();
	spoolDir = _spoolDir;
	workerId = _workerId;
	sessionTag = _sessionTag;
	targetPubkey = _targetPubkey;
	startOffset = _startOffset;
	rangeBits = _rangeBits;
	dpBits = _dpBits;
	flushRecords = std::max<u32>(1000, _flushRecords);
	flushSec = std::max<u32>(1, _flushSec);
	maxQueueRecords = std::max<size_t>(static_cast<size_t>(flushRecords) * 6ull, 200000ull);
	chunkSeq = 1;
	writtenRecords = 0;
	writtenFiles = 0;
	droppedRecords = 0;
	{
		std::lock_guard<std::mutex> lk(mtx);
		queue.clear();
		chunk.clear();
		stopRequested = false;
		isRunning = false;
	}
	if (!EnsureSpoolDir())
		return false;
	writerThread = std::thread(&WildSpoolWriter::WriterLoop, this);
	{
		std::lock_guard<std::mutex> lk(mtx);
		isRunning = true;
	}
	return true;
}

void WildSpoolWriter::Enqueue(const u8* xPrefix, const u8* distanceRaw24, u8 dpType)
{
	if (!xPrefix || !distanceRaw24)
		return;
	RawRecord rec;
	memcpy(rec.xPrefix, xPrefix, sizeof(rec.xPrefix));
	memcpy(rec.distanceRaw24, distanceRaw24, sizeof(rec.distanceRaw24));
	rec.dpType = dpType;
	rec.flags = 0;
	rec.reserved = 0;

	std::unique_lock<std::mutex> lk(mtx);
	if (!isRunning || stopRequested)
	{
		droppedRecords.fetch_add(1);
		return;
	}
	while (queue.size() >= maxQueueRecords && !stopRequested)
		cvSpace.wait_for(lk, std::chrono::milliseconds(100));
	if (stopRequested)
	{
		droppedRecords.fetch_add(1);
		return;
	}
	queue.push_back(rec);
	cvData.notify_one();
}

void WildSpoolWriter::StopAndFlush()
{
	{
		std::lock_guard<std::mutex> lk(mtx);
		if (!isRunning)
			return;
		stopRequested = true;
	}
	cvData.notify_all();
	cvSpace.notify_all();
	if (writerThread.joinable())
		writerThread.join();
	{
		std::lock_guard<std::mutex> lk(mtx);
		isRunning = false;
	}
}

u64 WildSpoolWriter::GetWrittenRecords() const
{
	return writtenRecords.load();
}

u64 WildSpoolWriter::GetWrittenFiles() const
{
	return writtenFiles.load();
}

u64 WildSpoolWriter::GetDroppedRecords() const
{
	return droppedRecords.load();
}

u64 WildSpoolWriter::GetPendingRecords() const
{
	std::lock_guard<std::mutex> lk(mtx);
	return static_cast<u64>(queue.size() + chunk.size());
}

bool WildSpoolWriter::EnsureSpoolDir()
{
	return EnsureDirRecursive(spoolDir);
}

void WildSpoolWriter::WriterLoop()
{
	u64 lastFlushMs = GetTickCount64();
	while (1)
	{
		std::vector<RawRecord> toWrite;
		{
			std::unique_lock<std::mutex> lk(mtx);
			cvData.wait_for(lk, std::chrono::milliseconds(200), [&]() { return stopRequested || !queue.empty(); });
			while (!queue.empty() && chunk.size() < flushRecords)
			{
				chunk.push_back(queue.front());
				queue.pop_front();
			}
			cvSpace.notify_all();

			u64 nowMs = GetTickCount64();
			bool shouldFlush = false;
			if (!chunk.empty())
			{
				if (chunk.size() >= flushRecords)
					shouldFlush = true;
				else if (nowMs - lastFlushMs >= static_cast<u64>(flushSec) * 1000ull)
					shouldFlush = true;
				else if (stopRequested && queue.empty())
					shouldFlush = true;
			}
			if (stopRequested && queue.empty() && chunk.empty())
				break;
			if (!shouldFlush)
				continue;
			toWrite.swap(chunk);
		}

		if (!toWrite.empty())
		{
			if (WriteChunk(toWrite))
			{
				writtenRecords.fetch_add(static_cast<u64>(toWrite.size()));
				writtenFiles.fetch_add(1);
			}
			else
			{
				droppedRecords.fetch_add(static_cast<u64>(toWrite.size()));
			}
			lastFlushMs = GetTickCount64();
		}
	}
}

bool WildSpoolWriter::WriteChunk(const std::vector<RawRecord>& records)
{
	if (records.empty())
		return true;

	char sidBuf[8];
	char seqBuf[16];
	char tsBuf[16];
	char rndBuf[16];
	u64 createdMs = UnixTimeMs();
	u64 seq = chunkSeq++;
	u32 entropy = static_cast<u32>((createdMs ^ (seq << 13) ^ static_cast<u64>(rand())) & 0xFFFFFFFFu);
	u32 sid = Fnv1a32(workerId + "|" + sessionTag) & 0xFFFFu;
	snprintf(sidBuf, sizeof(sidBuf), "%04X", static_cast<unsigned int>(sid));
	snprintf(seqBuf, sizeof(seqBuf), "%08llX", static_cast<unsigned long long>(seq & 0xFFFFFFFFull));
	snprintf(tsBuf, sizeof(tsBuf), "%08llX", static_cast<unsigned long long>(createdMs & 0xFFFFFFFFull));
	snprintf(rndBuf, sizeof(rndBuf), "%08X", static_cast<unsigned int>(entropy));
	std::string stem = "WDP." + std::string(sidBuf) + "." + std::string(seqBuf) + "." + std::string(tsBuf) + "." + std::string(rndBuf);
	std::string finalPath = spoolDir + "/" + stem + ".wdp";
	std::string partPath = finalPath + ".part";

	WdpHeader header;
	memset(&header, 0, sizeof(header));
	memcpy(header.magic, "WDP1", 4);
	header.version = static_cast<u16>(WDP_VERSION);
	header.headerSize = static_cast<u16>(sizeof(header));
	header.rangeBits = static_cast<u16>(rangeBits);
	header.dpBits = static_cast<u16>(dpBits);
	CopyTextField(header.workerId, sizeof(header.workerId), workerId);
	CopyTextField(header.sessionTag, sizeof(header.sessionTag), sessionTag);
	CopyTextField(header.targetPubkey, sizeof(header.targetPubkey), targetPubkey);
	CopyTextField(header.startOffset, sizeof(header.startOffset), startOffset);
	header.chunkSeq = seq;
	header.createdUnixMs = createdMs;
	header.recordCount = static_cast<u32>(records.size());

	const u8* bodyPtr = reinterpret_cast<const u8*>(records.data());
	size_t bodySize = records.size() * sizeof(RawRecord);
	header.bodyCrc32 = Crc32Update(0, bodyPtr, bodySize);

	WdpFooter footer;
	memset(&footer, 0, sizeof(footer));
	memcpy(footer.endMagic, "WDP1END", 7);
	u32 fileCrc = 0;
	fileCrc = Crc32Update(fileCrc, reinterpret_cast<const u8*>(&header), sizeof(header));
	fileCrc = Crc32Update(fileCrc, bodyPtr, bodySize);
	fileCrc = Crc32Update(fileCrc, reinterpret_cast<const u8*>(footer.endMagic), sizeof(footer.endMagic));
	footer.fileCrc32 = fileCrc;

	FILE* fp = fopen(partPath.c_str(), "wb");
	if (!fp)
		return false;
	bool ok = true;
	if (fwrite(&header, 1, sizeof(header), fp) != sizeof(header))
		ok = false;
	if (ok && fwrite(bodyPtr, 1, bodySize, fp) != bodySize)
		ok = false;
	if (ok && fwrite(&footer, 1, sizeof(footer), fp) != sizeof(footer))
		ok = false;
	if (ok && fflush(fp) != 0)
		ok = false;
#ifdef _WIN32
	if (ok && _commit(_fileno(fp)) != 0)
		ok = false;
#else
	if (ok && fsync(fileno(fp)) != 0)
		ok = false;
#endif
	fclose(fp);
	if (!ok)
	{
		remove(partPath.c_str());
		return false;
	}
	if (rename(partPath.c_str(), finalPath.c_str()) != 0)
	{
		remove(partPath.c_str());
		return false;
	}
	return true;
}
