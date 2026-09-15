#include "graphics/host_gpu/perfStats.h"

#include "common/profiler.h"
#include "common/timer.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::PerfStats {

namespace {

constexpr auto CounterCount  = static_cast<size_t>(Counter::Count);
constexpr auto DurationCount = static_cast<size_t>(Duration::Count);

std::array<std::atomic<uint64_t>, CounterCount>  g_counters {};
std::array<std::atomic<uint64_t>, DurationCount> g_durations {};

bool ConsoleEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PERF_STATS");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

} // namespace

void Add(Counter counter, uint64_t value) noexcept {
	g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

void AddTicks(Duration duration, uint64_t ticks) noexcept {
	g_durations[static_cast<size_t>(duration)].fetch_add(ticks, std::memory_order_relaxed);
}

uint64_t Now() noexcept {
	return Common::Timer::QueryPerformanceCounter();
}

void OnPresentedFrame() noexcept {
	const bool tracy_connected = tracy::ProfilerAvailable();
	if (tracy_connected) {
		FrameMark;
	}

	static uint64_t interval_start = Now();
	static uint64_t frames         = 0;
	frames++;
	const auto now       = Now();
	const auto frequency = Common::Timer::QueryPerformanceFrequency();
	if (frequency == 0 || now - interval_start < frequency) {
		return;
	}

	std::array<uint64_t, CounterCount> counters {};
	std::array<double, DurationCount>  millis {};
	for (size_t i = 0; i < CounterCount; i++) {
		counters[i] = g_counters[i].exchange(0, std::memory_order_relaxed);
	}
	for (size_t i = 0; i < DurationCount; i++) {
		millis[i] = static_cast<double>(g_durations[i].exchange(0, std::memory_order_relaxed)) *
		            1000.0 / static_cast<double>(frequency);
	}
	const double seconds =
	    static_cast<double>(now - interval_start) / static_cast<double>(frequency);
	const double fps = static_cast<double>(frames) / seconds;
	interval_start   = now;
	frames           = 0;

	const auto count = [&counters](Counter counter) {
		return static_cast<unsigned long long>(counters[static_cast<size_t>(counter)]);
	};
	const auto ms = [&millis](Duration duration) { return millis[static_cast<size_t>(duration)]; };

	if (tracy_connected) {
		TracyPlot("fps", fps);
		TracyPlot("draws", static_cast<int64_t>(count(Counter::Draws)));
		TracyPlot("dispatches", static_cast<int64_t>(count(Counter::Dispatches)));
		TracyPlot("submits", static_cast<int64_t>(count(Counter::Submits)));
		TracyPlot("read faults", static_cast<int64_t>(count(Counter::ReadFaults)));
		TracyPlot("write faults", static_cast<int64_t>(count(Counter::WriteFaults)));
		TracyPlot("cpu readbacks", static_cast<int64_t>(count(Counter::CpuReadbacks)));
		TracyPlot("shader compiles", static_cast<int64_t>(count(Counter::ShaderCompiles)));
		TracyPlot("pipeline compiles", static_cast<int64_t>(count(Counter::PipelineCompiles)));
		TracyPlot("bda walks", static_cast<int64_t>(count(Counter::BdaWalks)));
		TracyPlot("bda skips", static_cast<int64_t>(count(Counter::BdaSkips)));
		TracyPlot("materialize hits", static_cast<int64_t>(count(Counter::MaterializeHits)));
		TracyPlot("materialize misses", static_cast<int64_t>(count(Counter::MaterializeMisses)));
		TracyPlot("gpu wait ms", ms(Duration::GpuWait));
		TracyPlot("sync command ms", ms(Duration::SyncCommand));
		TracyPlot("flip wait ms", ms(Duration::FlipWait));
		TracyPlot("shader compile ms", ms(Duration::ShaderCompile));
		TracyPlot("pipeline compile ms", ms(Duration::PipelineCompile));
	}

	if (ConsoleEnabled()) {
		std::printf("PerfStats: interval=%.2fs fps=%.1f draws=%llu dispatches=%llu submits=%llu "
		            "read_faults=%llu write_faults=%llu readbacks=%llu shaders=%llu/%.0fms "
		            "pipelines=%llu/%.0fms bda=%llu/%llu mat=%llu/%llu gpu_wait=%.0fms "
		            "sync_cmd=%.0fms flip_wait=%.0fms\n",
		            seconds, fps, count(Counter::Draws), count(Counter::Dispatches),
		            count(Counter::Submits), count(Counter::ReadFaults), count(Counter::WriteFaults),
		            count(Counter::CpuReadbacks), count(Counter::ShaderCompiles),
		            ms(Duration::ShaderCompile), count(Counter::PipelineCompiles),
		            ms(Duration::PipelineCompile), count(Counter::BdaWalks), count(Counter::BdaSkips),
		            count(Counter::MaterializeHits), count(Counter::MaterializeMisses),
		            ms(Duration::GpuWait), ms(Duration::SyncCommand), ms(Duration::FlipWait));
		std::fflush(stdout);
	}
}

} // namespace Libs::Graphics::PerfStats
