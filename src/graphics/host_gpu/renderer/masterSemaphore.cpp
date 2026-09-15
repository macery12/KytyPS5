#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/perfStats.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::RecordSubmission(uint64_t tick, uint32_t op, uint64_t submit_id,
                                       uint32_t arg0, uint32_t arg1, uint32_t arg2,
                                       uint32_t arg3, uint64_t arg4, const uint64_t* shaders,
                                       uint32_t shader_count) {
	std::lock_guard lock(m_history_mutex);
	auto& entry = m_history[m_history_next++ % HistorySize];
	entry       = {tick, submit_id, arg4, op, {arg0, arg1, arg2, arg3}};
	entry.shader_count = std::min<uint32_t>(shader_count, MaxShaders);
	std::copy_n(shaders, entry.shader_count, entry.shaders.begin());
}

void MasterSemaphore::DumpRecentSubmissions(uint64_t known_tick) {
	std::lock_guard lock(m_history_mutex);
	const auto count = std::min(m_history_next, HistorySize);
	for (size_t i = 0; i < count; ++i) {
		const auto& entry = m_history[(m_history_next - count + i) % HistorySize];
		// Only the few completed submissions before the GPU stopped are interesting; the hang
		// is in the first incomplete one, which can be far behind the newest submission.
		if (entry.tick + 4 <= known_tick) {
			continue;
		}
		std::printf("GPU timeline recent submit: tick=%" PRIu64 " completed=%u op=%u "
		            "guest_submit=%" PRIu64 " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
		            entry.tick, entry.tick <= known_tick ? 1u : 0u, entry.op,
		            entry.submit_id, entry.args[0], entry.args[1], entry.args[2],
		            entry.args[3], entry.arg4);
		for (uint32_t s = 0; s < entry.shader_count; ++s) {
			std::printf("    shader 0x%016" PRIx64 "\n", entry.shaders[s]);
		}
	}
	std::fflush(stdout);
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	const auto last_issued_tick = CurrentTick() - 1;
	if (result != vk::Result::eSuccess) {
		EXIT("vkGetSemaphoreCounterValue failed: result=%d last_issued=%" PRIu64
		     " known_gpu=%" PRIu64 "\n",
		     static_cast<int>(result), last_issued_tick, KnownGpuTick());
	}
	if (counter > last_issued_tick) {
		uint64_t   repeated_counter = 0;
		const auto repeated_result  =
		    m_graphics.device.getSemaphoreCounterValue(m_semaphore, &repeated_counter);
		vk::SemaphoreWaitInfo poll_info {};
		poll_info.semaphoreCount = 1;
		poll_info.pSemaphores    = &m_semaphore;
		poll_info.pValues        = &last_issued_tick;
		const auto poll_result   = m_graphics.device.waitSemaphores(&poll_info, 0);
		if (counter == UINT64_MAX) {
			std::printf("GPU device lost (driver reset/TDR): the first submission with "
			            "completed=0 below most likely contains the hanging shader\n");
		}
		DumpRecentSubmissions(KnownGpuTick());
		EXIT("GPU timeline counter exceeded the last issued tick: counter=%" PRIu64
		     " last_issued=%" PRIu64 " known_gpu=%" PRIu64 " repeated_counter=%" PRIu64
		     " repeated_result=%d poll_result=%d\n",
		     counter, last_issued_tick, KnownGpuTick(), repeated_counter,
		     static_cast<int>(repeated_result), static_cast<int>(poll_result));
	}

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	KYTY_PROFILER_BLOCK("MasterSemaphore::Wait", profiler::colors::RedA100);
	PerfStats::ScopedDuration wait_time(PerfStats::Duration::GpuWait);
	const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

} // namespace Libs::Graphics
