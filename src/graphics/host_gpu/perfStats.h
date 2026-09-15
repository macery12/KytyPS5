#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PERFSTATS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PERFSTATS_H_

#include <cstdint>

namespace Libs::Graphics::PerfStats {

// Host renderer activity summed over roughly one second of presented frames. KYTY_PERF_STATS=1
// prints one console line per interval; a connected Tracy client receives the same values as plots
// together with a frame mark per presented frame. Counting is a relaxed atomic add, cheap enough
// for every draw.
enum class Counter : uint32_t {
	Draws,
	Dispatches,
	Submits,
	ReadFaults,
	WriteFaults,
	CpuReadbacks,
	ShaderCompiles,
	PipelineCompiles,
	Count
};

// Wall time summed over every thread, so a total can exceed the interval.
enum class Duration : uint32_t {
	GpuWait,
	SyncCommand,
	FlipWait,
	ShaderCompile,
	PipelineCompile,
	Count
};

void     Add(Counter counter, uint64_t value = 1) noexcept;
void     AddTicks(Duration duration, uint64_t ticks) noexcept;
uint64_t Now() noexcept;

// Called once per presented host frame.
void OnPresentedFrame() noexcept;

class ScopedDuration {
public:
	explicit ScopedDuration(Duration duration) noexcept: m_duration(duration), m_start(Now()) {}
	~ScopedDuration() { AddTicks(m_duration, Now() - m_start); }

	ScopedDuration(const ScopedDuration&)            = delete;
	ScopedDuration& operator=(const ScopedDuration&) = delete;

private:
	Duration m_duration;
	uint64_t m_start;
};

} // namespace Libs::Graphics::PerfStats

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_PERFSTATS_H_ */
