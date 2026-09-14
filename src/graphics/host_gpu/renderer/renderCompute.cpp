#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// Copy one full-size image to a host-visible buffer, then write it after that GPU tick finishes.
// The caller selects the exact shader/resource so this stays outside ordinary rendering.
static bool QueueUiImageDump(RenderContext& context, Image& image, uint64_t frame,
                             const char* name, const char* extension) {
	const auto width = image.backing.extent.width;
	const auto height = image.backing.extent.height;
	const uint64_t size = static_cast<uint64_t>(width) * height * 4u;
	auto& download = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Download);
	// A diagnostic must never wait for a staging region still owned by this GPU tick.
	const auto [mapped, offset] = download.Map(size, 4, false);
	if (mapped == nullptr) {
		std::printf("UI trace: %s readback buffer unavailable\n", name);
		std::fflush(stdout);
		return false;
	}
	download.Commit();
	vk::BufferImageCopy copy {};
	copy.bufferOffset = offset;
	copy.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
	copy.imageSubresource.layerCount = 1;
	copy.imageExtent = {width, height, 1};
	const std::array copies {copy};
	image.Download(copies, download.Handle(), offset, size);
	vk::BufferMemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer = download.Handle();
	barrier.offset = offset;
	barrier.size = size;
	context.GetCommandScheduler().Current().Handle().pipelineBarrier(
	    vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {},
	    0, nullptr, 1, &barrier, 0, nullptr);
	const auto address = image.info.data.address;
	const auto format = image.backing.format;
	const std::string path = "_UiTrace/" + std::string(name) + "_" + std::to_string(frame) + extension;
	context.GetCommandScheduler().DeferPriorityOperation(
	    [&download, mapped, offset, size, frame, address, format, width, height, path] {
		    download.Invalidate(offset, size);
		    std::error_code error;
		    std::filesystem::create_directories("_UiTrace", error);
		    if (error) {
			    std::printf("UI trace: failed to create _UiTrace: %s\n",
			                error.message().c_str());
			    return;
		    }
		    auto* file = std::fopen(path.c_str(), "wb");
		    if (file == nullptr) {
			    std::printf("UI trace: failed to open image dump %s\n", path.c_str());
			    return;
		    }
		    const auto written = std::fwrite(mapped, 1, size, file);
		    std::fclose(file);
		    uint64_t nonzero_alpha = 0;
		    const bool packed10 = format == vk::Format::eA2B10G10R10UnormPack32 ||
		                          format == vk::Format::eA2R10G10B10UnormPack32;
		    for (uint64_t i = 3; i < size; i += 4) {
			    nonzero_alpha += packed10 ? (mapped[i] & 0xc0u) != 0 : mapped[i] != 0;
		    }
		    std::printf("UI trace: image dump=%s frame=%llu guest=0x%llx %ux%u"
		                " format=%u bytes=%zu nonzero_alpha=%llu/%llu\n",
		                path.c_str(), static_cast<unsigned long long>(frame),
		                static_cast<unsigned long long>(address), width, height,
		                static_cast<uint32_t>(format), written,
		                static_cast<unsigned long long>(nonzero_alpha),
		                static_cast<unsigned long long>(size / 4));
		    std::fflush(stdout);
	    });
	std::printf("UI trace: queued %s readback at frame=%llu\n", name,
	            static_cast<unsigned long long>(frame));
	std::fflush(stdout);
	return true;
}

// Capture the compositor's late-sample UV scales (40/44) and gate word (148)
// from the exact bound Vulkan buffer. The descriptor may expose only 152 bytes.
static bool QueueUiCompositorParameterDump(RenderContext& context,
                                           const PreparedBindings& bindings,
                                           uint64_t frame, uint32_t offset_dword) {
	if (bindings.buffer_sources.size() < 2 || bindings.buffers.size() < 2 ||
	    bindings.shader_data.size() <= offset_dword)
		return false;
	const auto& source = bindings.buffer_sources[1];
	const auto& bound = bindings.buffers[1];
	constexpr uint64_t capture_size = 12;
	constexpr uint64_t required_size = 152;
	if (source.address == 0 || source.size < required_size || bound.buffer == nullptr)
		return false;
	const uint32_t adjustment = (bindings.shader_data[offset_dword] >> 8u) & 0xffu;
	if (bound.range < adjustment + required_size) return false;
	const uint64_t source_offset = bound.offset + adjustment;
	auto& download = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Download);
	const auto [mapped, offset] = download.Map(capture_size, 4, false);
	if (mapped == nullptr) return false;
	download.Commit();
	std::array<uint8_t, capture_size> guest_bytes {};
	const bool guest_uv_clean = Libs::LibKernel::Memory::TryReadGpuCleanBacking(
	    source.address + 40, guest_bytes.data(), 8);
	const bool guest_flag_clean = Libs::LibKernel::Memory::TryReadGpuCleanBacking(
	    source.address + 148, guest_bytes.data() + 8, 4);
	auto command = context.GetCommandScheduler().Current().Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eHostWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer = bound.buffer;
	before.offset = source_offset + 40;
	before.size = required_size - 40;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands |
	                            vk::PipelineStageFlagBits::eHost,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before,
	                        0, nullptr);
	const std::array copies {vk::BufferCopy {source_offset + 40, offset, 8},
	                         vk::BufferCopy {source_offset + 148, offset + 8, 4}};
	command.copyBuffer(bound.buffer, download.Handle(),
	                   static_cast<uint32_t>(copies.size()), copies.data());
	vk::BufferMemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.buffer = download.Handle();
	after.offset = offset;
	after.size = capture_size;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &after,
	                        0, nullptr);
	context.GetCommandScheduler().DeferPriorityOperation(
	    [&download, mapped, offset, frame, capture_size, guest_uv_clean, guest_flag_clean,
	     guest_bytes,
	     address = source.address,
	     adjustment] {
		    download.Invalidate(offset, capture_size);
		    uint32_t gpu_word = 0;
		    uint32_t guest_word = 0;
		    float uv_x = 0.0f, uv_y = 0.0f;
		    std::memcpy(&gpu_word, mapped + 8, sizeof(gpu_word));
		    std::memcpy(&guest_word, guest_bytes.data() + 8, sizeof(guest_word));
		    std::memcpy(&uv_x, mapped, sizeof(uv_x));
		    std::memcpy(&uv_y, mapped + 4, sizeof(uv_y));
		    std::error_code error;
		    std::filesystem::create_directories("_UiTrace", error);
		    const auto path = "_UiTrace/compositor_params_" + std::to_string(frame) + ".bin";
		    bool saved = false;
		    if (!error) {
			    if (auto* file = std::fopen(path.c_str(), "wb")) {
				    saved = std::fwrite(mapped, 1, capture_size, file) == capture_size;
				    std::fclose(file);
			    }
		    }
		    std::printf("UI trace: compositor flag frame=%llu buffer[1]=0x%llx"
		                " adjust=%u offset=148 gpu=0x%08x bit4=%u"
		                " guest_clean=%u guest=0x%08x uv_scale=%g,%g"
		                " guest_uv_clean=%u params=%s\n",
		                static_cast<unsigned long long>(frame),
		                static_cast<unsigned long long>(address), adjustment, gpu_word,
		                (gpu_word >> 4u) & 1u, guest_flag_clean ? 1u : 0u, guest_word,
		                static_cast<double>(uv_x), static_cast<double>(uv_y),
		                guest_uv_clean ? 1u : 0u,
		                saved ? path.c_str() : "unavailable");
		    std::fflush(stdout);
	    });
	return true;
}

// The compositor's IMAGE_LOAD at 0x7ec is enabled by a bit read from its
// mask buffer (SPIR-V buffer resource 4 in the captured shader). Capture both
// the GPU-visible bytes and clean guest backing to separate bad mask contents
// from a stale or incorrectly bound host buffer.
static bool QueueUiCompositorMaskDump(RenderContext& context,
                                      const PreparedBindings& bindings,
                                      uint64_t frame, uint32_t offset_dword,
                                      uint32_t first_use_pc) {
	constexpr uint32_t mask_resource = 4;
	if (bindings.buffer_sources.size() <= mask_resource ||
	    bindings.buffers.size() <= mask_resource ||
	    bindings.shader_data.size() <= offset_dword + mask_resource / 4u)
		return false;
	const auto& source = bindings.buffer_sources[mask_resource];
	const auto& bound = bindings.buffers[mask_resource];
	const uint32_t adjustment =
	    (bindings.shader_data[offset_dword + mask_resource / 4u] >>
	     ((mask_resource % 4u) * 8u)) & 0xffu;
	if (source.address == 0 || source.size < 4 || bound.buffer == nullptr ||
	    bound.range <= adjustment)
		return false;
	const uint64_t size = std::min<uint64_t>({source.size, bound.range - adjustment, 64u * 1024u}) & ~3ull;
	if (size == 0) return false;
	const uint64_t source_offset = bound.offset + adjustment;
	auto& download = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Download);
	const auto [mapped, offset] = download.Map(size, 4, false);
	if (mapped == nullptr) return false;
	download.Commit();
	std::vector<uint8_t> guest_bytes(size);
	const bool guest_clean = Libs::LibKernel::Memory::TryReadGpuCleanBacking(
	    source.address, guest_bytes.data(), size);
	auto command = context.GetCommandScheduler().Current().Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eHostWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer = bound.buffer;
	before.offset = source_offset;
	before.size = size;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands |
	                            vk::PipelineStageFlagBits::eHost,
	                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before,
	                        0, nullptr);
	const vk::BufferCopy copy {source_offset, offset, size};
	command.copyBuffer(bound.buffer, download.Handle(), 1, &copy);
	vk::BufferMemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	after.buffer = download.Handle();
	after.offset = offset;
	after.size = size;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                        vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &after,
	                        0, nullptr);
	context.GetCommandScheduler().DeferPriorityOperation(
	    [&download, mapped, offset, size, frame, guest_clean, guest_bytes,
	    address = source.address, source_size = source.size, first_use_pc] {
		    download.Invalidate(offset, size);
		    uint64_t gpu_nonzero = 0;
		    uint64_t guest_nonzero = 0;
		    for (uint64_t i = 0; i < size; i += 4) {
			    uint32_t gpu_word = 0, guest_word = 0;
			    std::memcpy(&gpu_word, mapped + i, 4);
			    std::memcpy(&guest_word, guest_bytes.data() + i, 4);
			    gpu_nonzero += gpu_word != 0;
			    guest_nonzero += guest_word != 0;
		    }
		    std::error_code error;
		    std::filesystem::create_directories("_UiTrace", error);
		    const auto path = "_UiTrace/compositor_mask_" + std::to_string(frame) + ".bin";
		    bool saved = false;
		    if (!error) {
			    if (auto* file = std::fopen(path.c_str(), "wb")) {
				    saved = std::fwrite(mapped, 1, size, file) == size;
				    std::fclose(file);
			    }
		    }
		    std::printf("UI trace: compositor mask frame=%llu buffer[4]=0x%llx"
		                " first_use_pc=0x%08x"
		                " source_size=%llu bytes=%llu gpu_nonzero=%llu/%llu"
		                " guest_clean=%u guest_nonzero=%llu equal=%u dump=%s\n",
		                static_cast<unsigned long long>(frame),
		                static_cast<unsigned long long>(address), first_use_pc,
		                static_cast<unsigned long long>(source_size),
		                static_cast<unsigned long long>(size),
		                static_cast<unsigned long long>(gpu_nonzero),
		                static_cast<unsigned long long>(size / 4), guest_clean ? 1u : 0u,
		                static_cast<unsigned long long>(guest_nonzero),
		                guest_clean && std::memcmp(mapped, guest_bytes.data(), size) == 0 ? 1u : 0u,
		                saved ? path.c_str() : "unavailable");
		    std::fflush(stdout);
	    });
	return true;
}

// KYTY_TRACE_NAN_CS=<hash>[,<hash>...]: read back the storage buffers of the listed compute
// shaders before and after each dispatch and report float words that become NaN/Inf. Used to find
// the first dispatch that poisons GPU-simulated vertex data (NHL 26 exploding skinned meshes).
struct NanTraceCapture {
	uint32_t index   = 0;
	bool     written = false;
	uint64_t address = 0;
	uint64_t size    = 0;
	uint8_t* mapped  = nullptr;
	uint64_t offset  = 0;
	// Guest memory at the same range, read when the guest backing is still clean. Tells whether
	// NaN words were uploaded from the guest or exist only in the host copy.
	bool                 guest_clean = false;
	std::vector<uint8_t> guest;
};

// KYTY_TRACE_NAN_CS=* traces every compute shader.
static bool NanTraceEnabledFor(uint64_t hash) {
	static bool                        all    = false;
	static const std::vector<uint64_t> hashes = [] {
		std::vector<uint64_t> list;
		const char*           value = std::getenv("KYTY_TRACE_NAN_CS");
		if (value == nullptr) {
			return list;
		}
		if (std::strcmp(value, "*") == 0) {
			all = true;
			std::printf("NanTrace: armed for all compute shaders\n");
			std::fflush(stdout);
			return list;
		}
		std::string text(value);
		size_t      start = 0;
		while (start < text.size()) {
			const auto end = text.find(',', start);
			const auto item = text.substr(start, end == std::string::npos ? std::string::npos
			                                                              : end - start);
			if (!item.empty()) {
				list.push_back(std::strtoull(item.c_str(), nullptr, 16));
			}
			if (end == std::string::npos) {
				break;
			}
			start = end + 1;
		}
		std::printf("NanTrace: armed for %zu compute shader(s)\n", list.size());
		std::fflush(stdout);
		return list;
	}();
	return all || std::find(hashes.begin(), hashes.end(), hash) != hashes.end();
}

static std::vector<NanTraceCapture> QueueNanTraceCopies(RenderContext&           context,
                                                        const PreparedBindings&  bindings,
                                                        const std::vector<uint8_t>& written,
                                                        uint32_t offset_dword,
                                                        uint32_t offset_count, bool written_only) {
	std::vector<NanTraceCapture> captures;
	auto& download = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Download);
	auto  command  = context.GetCommandScheduler().Current().Handle();
	const auto count = std::min({written.size(), bindings.buffers.size(),
	                             bindings.buffer_sources.size()});
	for (uint32_t i = 0; i < count; ++i) {
		if (written_only && written[i] == 0) continue;
		const auto&    source     = bindings.buffer_sources[i];
		const auto&    bound      = bindings.buffers[i];
		const uint32_t adjustment = i < offset_count && offset_dword + i / 4u < bindings.shader_data.size()
		                                ? (bindings.shader_data[offset_dword + i / 4u] >>
		                                   ((i % 4u) * 8u)) & 0xffu
		                                : 0u;
		if (source.address == 0 || source.size < 4 || bound.buffer == nullptr ||
		    bound.range <= adjustment)
			continue;
		const uint64_t size =
		    std::min<uint64_t>({source.size, bound.range - adjustment, 256u * 1024u}) & ~3ull;
		if (size == 0) continue;
		const auto [mapped, offset] = download.Map(size, 4, false);
		if (mapped == nullptr) break;
		download.Commit();
		const uint64_t          source_offset = bound.offset + adjustment;
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryWrite | vk::AccessFlagBits::eHostWrite |
		                       vk::AccessFlagBits::eShaderWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = bound.buffer;
		before.offset              = source_offset;
		before.size                = size;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands |
		                            vk::PipelineStageFlagBits::eHost,
		                        vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
		                        nullptr);
		const vk::BufferCopy copy {source_offset, offset, size};
		command.copyBuffer(bound.buffer, download.Handle(), 1, &copy);
		vk::BufferMemoryBarrier after {};
		after.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask       = vk::AccessFlagBits::eHostRead;
		after.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		after.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		after.buffer              = download.Handle();
		after.offset              = offset;
		after.size                = size;
		command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                        vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1, &after, 0,
		                        nullptr);
		auto& capture = captures.emplace_back();
		capture.index   = i;
		capture.written = written[i] != 0;
		capture.address = source.address;
		capture.size    = size;
		capture.mapped  = mapped;
		capture.offset  = offset;
		if (!written_only) {
			capture.guest.resize(size);
			capture.guest_clean = Libs::LibKernel::Memory::TryReadGpuCleanBacking(
			    source.address, capture.guest.data(), size);
			if (!capture.guest_clean) capture.guest.clear();
		}
	}
	return captures;
}

static void QueueNanTraceReport(RenderContext& context, std::vector<NanTraceCapture> before,
                                std::vector<NanTraceCapture> after, uint64_t hash, uint64_t frame,
                                uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
	auto& download = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Download);
	context.GetCommandScheduler().DeferPriorityOperation(
	    [&download, before = std::move(before), after = std::move(after), hash, frame, groups_x,
	     groups_y, groups_z] {
		    struct Count {
			    uint64_t bad   = 0;
			    int64_t  first = -1;
			    std::string indices;
		    };
		    // 0xffffffff is a common integer sentinel, not a float, so it is not counted.
		    const auto scan = [](const uint8_t* bytes, uint64_t size) {
			    Count result;
			    for (uint64_t byte = 0; byte < size; byte += 4) {
				    uint32_t word = 0;
				    std::memcpy(&word, bytes + byte, 4);
				    if (((word >> 23u) & 0xffu) == 0xffu && word != 0xffffffffu) {
					    if (result.first < 0) result.first = static_cast<int64_t>(byte / 4);
					    if (result.bad < 4) {
						    result.indices += fmt::format("{}{}:{:08x}", result.bad ? "," : "",
						                                  byte / 4, word);
					    }
					    ++result.bad;
				    }
			    }
			    return result;
		    };
		    const auto count = [&](const NanTraceCapture& capture) {
			    download.Invalidate(capture.offset, capture.size);
			    return scan(capture.mapped, capture.size);
		    };
		    std::string inputs;
		    std::string outputs;
		    bool        poisoned = false;
		    for (const auto& capture: before) {
			    const auto pre = count(capture);
			    inputs += fmt::format(" {}:{}/{}", capture.index, pre.bad, capture.size / 4);
			    if (pre.bad != 0) {
				    const auto guest = capture.guest_clean
				                           ? fmt::format("{}", scan(capture.guest.data(),
				                                                    capture.guest.size()).bad)
				                           : std::string("dirty");
				    inputs += fmt::format("[@0x{:010x} guest={} {{{}}}]", capture.address, guest,
				                          pre.indices);
			    }
			    if (!capture.written) continue;
			    for (const auto& written: after) {
				    if (written.index != capture.index) continue;
				    const auto post = count(written);
				    poisoned |= post.bad > pre.bad;
				    outputs += fmt::format(" {}@0x{:010x}:{}->{} first={}", written.index,
				                           written.address, pre.bad, post.bad, post.first);
			    }
		    }
		    static std::mutex                             mutex;
		    static std::unordered_map<uint64_t, uint32_t> dispatches;
		    static uint32_t                               lines = 0;
		    std::lock_guard                               lock(mutex);
		    const auto                                    seen = dispatches[hash]++;
		    if (lines >= 2000 || (!poisoned && seen >= 3)) return;
		    ++lines;
		    std::printf("NanTrace: %s frame=%llu cs=0x%016llx groups=%ux%ux%u buffers(bad/words):%s |"
		                " outputs(before->after):%s\n",
		                poisoned ? "POISON" : "sample", static_cast<unsigned long long>(frame),
		                static_cast<unsigned long long>(hash), groups_x, groups_y, groups_z,
		                inputs.c_str(), outputs.c_str());
		    std::fflush(stdout);
	    });
}

static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer&          buffer) {
	const auto& program   = *input.stage.program;
	const auto& resources = input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	// KYTY_TRACE_DCC=1: report why a recognized uniform buffer fill is not consumed.
	const auto reject = [&](const char* reason) {
		static const bool trace = [] {
			const char* value = std::getenv("KYTY_TRACE_DCC");
			return value != nullptr && std::strcmp(value, "1") == 0;
		}();
		static std::atomic<uint32_t> count {0};
		if (trace && count.fetch_add(1, std::memory_order_relaxed) < 256) {
			std::printf("DccTrace: fill rejected (%s) cs=0x%016llx addr=0x%010llx records=%llu "
			            "stride=%u words=%u groups=%ux%ux%u threads=%ux%ux%u mode=0x%x\n",
			            reason, static_cast<unsigned long long>(input.stage.program->shader_hash),
			            static_cast<unsigned long long>(descriptor.Base48()),
			            static_cast<unsigned long long>(descriptor.NumRecords()),
			            static_cast<uint32_t>(descriptor.Stride()), fill.words, group_x, group_y,
			            group_z, input.threads_num[0], input.threads_num[1], input.threads_num[2],
			            mode);
			std::fflush(stdout);
		}
		return false;
	};
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.Base48() == 0) {
		return reject("descriptor");
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return reject("dispatch shape");
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	if (invocations != descriptor.NumRecords() || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return reject("coverage");
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return reject("aliased source");
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

bool RenderExecutor::TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
                                                CommandBuffer& command, uint32_t group_x,
                                                uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = *input.stage.program;
	const auto& resources = input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear)) {
		// Track deferred DCC state while the original dispatch writes the metadata allocation.
		cache.TrackDccFill(descriptor.Base48(), size, packed_clear);
		static std::atomic<uint32_t> logged_metadata_clears {0};
		if (logged_metadata_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: metadata fill shader=0x%016" PRIx64
			     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
			     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
		}
		return false;
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute image clear shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
	}
	return true;
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, CommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode) {
	EXIT_IF(buffer.IsInvalid());
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	Common::LockGuard lock(m_context.GetMutex());
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	if (!ShaderAddressValid(sh_ctx.GetCs().cs_regs.data_addr)) {
		return;
	}
	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo input_info {};
	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	const auto compute_program =
	    m_context.GetPipelineCache().GetComputeProgram(cs_regs, sh_regs, input_info);
	// Diagnostic filters: comma-separated lists of guest CS hashes / code addresses to skip.
	struct SkipList {
		uint64_t values[16] {};
		uint32_t count = 0;
		[[nodiscard]] bool Contains(uint64_t v) const {
			for (uint32_t i = 0; i < count; i++) {
				if (values[i] == v) return true;
			}
			return false;
		}
	};
	const auto parse_skip_list = [](const char* name) {
		SkipList list;
		const char* value = std::getenv(name);
		while (value != nullptr && *value != '\0' && list.count < 16) {
			char* end = nullptr;
			const uint64_t v = std::strtoull(value, &end, 0);
			if (end == value || v == 0) {
				std::printf("Diagnostic: ignoring invalid %s entry\n", name);
				break;
			}
			list.values[list.count++] = v;
			std::printf("Diagnostic: guest CS %s filter armed for 0x%016" PRIx64 "\n", name, v);
			value = (*end == ',') ? end + 1 : end;
		}
		std::fflush(stdout);
		return list;
	};
	// Default when KYTY_SKIP_CS_HASH is unset: NHL 26 CS 0873b3a2bce038b2 crashes the AMD driver in
	// vkCreateComputePipelines (amdvlk64.dll+0x2b5ab9c). A no-op replacement ran fine, and this
	// skip happens before GetComputePipeline, so skipping is equivalent. Set the variable
	// (e.g. to a harmless value) to replace this default list.
	static const SkipList skip_cs_hashes = [&] {
		if (std::getenv("KYTY_SKIP_CS_HASH") != nullptr) {
			return parse_skip_list("KYTY_SKIP_CS_HASH");
		}
		SkipList defaults;
		defaults.values[defaults.count++] = 0x0873b3a2bce038b2ull;
		return defaults;
	}();
	static const SkipList skip_cs_addresses = parse_skip_list("KYTY_SKIP_CS_ADDRESS");
	if (skip_cs_hashes.Contains(input_info.stage.program->shader_hash) ||
	    skip_cs_addresses.Contains(cs_regs.cs_regs.data_addr)) {
		static std::atomic<uint32_t> skipped {0};
		static const bool trace_startup = [] {
			const char* value = std::getenv("KYTY_TRACE_STARTUP");
			return value != nullptr && std::strcmp(value, "1") == 0;
		}();
		const uint32_t skip_count = skipped.fetch_add(1, std::memory_order_relaxed) + 1;
		if (skip_count == 1) {
			std::printf("Skipping guest CS hash=0x%016" PRIx64 " address=0x%016" PRIx64
			            " groups=%ux%ux%u (KYTY_SKIP_CS diagnostic)\n",
			            input_info.stage.program->shader_hash, cs_regs.cs_regs.data_addr,
			            thread_group_x, thread_group_y,
			            thread_group_z);
			std::fflush(stdout);
			if (trace_startup) {
				const auto& program   = *input_info.stage.program;
				const auto& resources = input_info.stage.resources;
				std::printf("Startup trace: skipped CS local=%ux%ux%u wave=%u buffers=%zu images=%zu\n",
				            input_info.threads_num[0], input_info.threads_num[1],
				            input_info.threads_num[2], input_info.wave_size,
				            program.info.buffers.size(), program.info.images.size());
				for (size_t i = 0; i < program.info.buffers.size(); i++) {
					const auto& resource = program.info.buffers[i];
					const auto desc = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
					std::printf("Startup trace: skipped CS buffer[%zu] source=%u written=%u "
					            "addr=0x%012" PRIx64 " stride=%u records=%u\n",
					            i, resource.source, resource.written ? 1u : 0u, desc.Base48(),
					            desc.Stride(), desc.NumRecords());
				}
				for (size_t i = 0; i < program.info.images.size(); i++) {
					const auto& image = program.info.images[i];
					const auto desc = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
					std::printf("Startup trace: skipped CS image[%zu] source=%u storage=%u written=%u "
					            "addr=0x%010" PRIx64 " type=%u format=%u extent=%ux%u\n",
					            i, image.source,
					            image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage
					                ? 1u : 0u,
					            image.written ? 1u : 0u, desc.Base40(), static_cast<uint32_t>(desc.Type()),
					            static_cast<uint32_t>(desc.Format()),
					            static_cast<uint32_t>(desc.Width5()) + 1u,
					            static_cast<uint32_t>(desc.Height5()) + 1u);
				}
				std::fflush(stdout);
			}
		} else if (trace_startup && (skip_count & (skip_count - 1u)) == 0) {
			std::printf("Startup trace: skipped CS dispatches=%u\n", skip_count);
			std::fflush(stdout);
		}
		return;
	}

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = *input_info.stage.program;
	const auto& resources = input_info.stage.resources;
	if (TryConsumeComputeMetaClear(input_info, buffer)) {
		ResetBindings();
		return;
	}
	if (TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y,
	                                thread_group_z, mode)) {
		ResetBindings();
		return;
	}
	const bool large_workgroup =
	    (input_info.threads_num[0] * input_info.threads_num[1] * input_info.threads_num[2] >= 512);
	const bool                   has_sampler = !program.info.samplers.empty();
	static std::atomic<uint32_t> dispatch_log_count {0};
	if ((large_workgroup || has_sampler) &&
	    dispatch_log_count.fetch_add(1, std::memory_order_relaxed) < 512) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x, thread_group_y,
		     thread_group_z, mode, input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(image.r128 ? r.LastLevel() : r.MaxMip()) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(resources.samplers[i]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	if (use_thread_dimensions) {
		auto groups_from_threads = [](uint32_t threads, uint32_t group_size) {
			return (threads == 0
			            ? 0u
			            : (threads + std::max(group_size, 1u) - 1u) / std::max(group_size, 1u));
		};

		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = groups_from_threads(thread_group_x, cs_regs.cs_regs.num_thread_x);
		thread_group_y       = groups_from_threads(thread_group_y, cs_regs.cs_regs.num_thread_y);
		thread_group_z       = groups_from_threads(thread_group_z, cs_regs.cs_regs.num_thread_z);

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}
	static const uint64_t ui_trace_start = [] {
		const char* value = std::getenv("KYTY_TRACE_UI");
		if (value == nullptr || *value == '\0') return UINT64_MAX;
		char*          end   = nullptr;
		const uint64_t frame = std::strtoull(value, &end, 10);
		return end != value && *end == '\0' && frame != 0 ? frame : UINT64_MAX;
	}();
	if (ui_trace_start != UINT64_MAX) {
		const auto frame = WindowGetPresentedFrameNum();
		if (frame >= ui_trace_start) {
			static std::mutex                             ui_trace_mutex;
			static std::unordered_map<uint64_t, uint32_t> seen;
			static std::unordered_map<uint64_t, bool>     input_logged;
			static std::atomic<uint32_t>                  logged {0};
			if (logged.load(std::memory_order_relaxed) < 128) {
				auto trace_output = [&](const char* kind, uint64_t address, uint64_t bytes,
				                        uint32_t width, uint32_t height) {
					if (address == 0) return;
					const uint64_t key = program.shader_hash ^ (address * 0x9e3779b97f4a7c15ull) ^
					                     (kind[0] == 'i' ? 1ull : 0ull);
					std::lock_guard trace_lock(ui_trace_mutex);
					if (logged.load(std::memory_order_relaxed) >= 128 || seen[key]++ >= 2) return;
					logged.fetch_add(1, std::memory_order_relaxed);
					std::printf("UI trace: compute frame=%llu cs=0x%016" PRIx64
					            " output=%s addr=0x%012" PRIx64 " bytes=%llu extent=%ux%u"
					            " groups=%ux%ux%u\n",
					            static_cast<unsigned long long>(frame), program.shader_hash, kind,
					            address, static_cast<unsigned long long>(bytes), width, height,
					            thread_group_x, thread_group_y, thread_group_z);
					if (!input_logged[program.shader_hash]) {
						input_logged[program.shader_hash] = true;
						uint32_t inputs                   = 0;
						for (size_t i = 0; i < program.info.images.size() && inputs < 8; ++i) {
							if (!program.info.images[i].read) continue;
							const auto desc =
							    DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
							const uint32_t w = static_cast<uint32_t>(desc.Width5()) + 1u;
							const uint32_t h = static_cast<uint32_t>(desc.Height5()) + 1u;
							if (static_cast<uint64_t>(w) * h < 1280ull * 720ull) continue;
							std::printf("UI trace: compute input cs=0x%016" PRIx64
							            " image[%zu]=0x%010" PRIx64 " extent=%ux%u written=%u\n",
							            program.shader_hash, i, desc.Base40(), w, h,
							            program.info.images[i].written ? 1u : 0u);
							++inputs;
						}
					}
					std::fflush(stdout);
				};
				for (size_t i = 0; i < program.info.images.size(); ++i) {
					const auto& image = program.info.images[i];
					if (!image.written ||
					    image.resource_class != ShaderRecompiler::IR::ImageResourceClass::Storage)
						continue;
					const auto desc =
					    DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
					const uint32_t width  = static_cast<uint32_t>(desc.Width5()) + 1u;
					const uint32_t height = static_cast<uint32_t>(desc.Height5()) + 1u;
					if (static_cast<uint64_t>(width) * height < 1280ull * 720ull) continue;
					trace_output("image", desc.Base40(), 0, width, height);
				}
				for (size_t i = 0; i < program.info.buffers.size(); ++i) {
					if (!program.info.buffers[i].written) continue;
					const auto desc =
					    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
					const uint64_t bytes = desc.GetSize();
					if (bytes < 4ull * 1024ull * 1024ull) continue;
					trace_output("buffer", desc.Base48(), bytes, 0, 0);
				}
			}
		}
	}

	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info, compute_program);
	auto bindings = PrepareBindings(input_info.stage);
	FindBuffers(bindings);
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	}
	RebindBuffers(bindings);
	RebindImages(bindings);
	// Capture the UI surface immediately after its draws and before the final compositor reads it.
	// This diagnostic is opt-in and takes one GPU-to-host copy for the entire run.
	static const bool dump_ui_target = [] {
		const char* value = std::getenv("KYTY_DUMP_UI_TARGET");
		const bool enabled = value != nullptr && std::strcmp(value, "1") == 0;
		if (enabled) {
			std::printf("UI trace: target dump armed\n");
			std::fflush(stdout);
		}
		return enabled;
	}();
	if (dump_ui_target && program.shader_hash == 0xd8ad62a44c60a9a3ull &&
	    ui_trace_start != UINT64_MAX && WindowGetPresentedFrameNum() >= ui_trace_start &&
	    bindings.images.size() > 3) {
		static std::atomic<bool> captured {false};
		const auto& binding = bindings.images[3];
		auto& image = m_context.GetTextureCache().GetImage(binding.image_id);
		const auto width = image.backing.extent.width;
		const auto height = image.backing.extent.height;
		if (width != 1920 || height != 1080 ||
		    image.backing.format != vk::Format::eR8G8B8A8Unorm || !image.usage.render_target) {
			static std::atomic<bool> warned {false};
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				std::printf("UI trace: target dump skipped guest=0x%llx %ux%u"
				            " format=%u target=%u\n",
				            static_cast<unsigned long long>(image.info.data.address), width, height,
				            static_cast<uint32_t>(image.backing.format),
				            image.usage.render_target ? 1u : 0u);
				std::fflush(stdout);
			}
		} else if (!captured.exchange(true, std::memory_order_relaxed)) {
			if (!QueueUiImageDump(m_context, image, WindowGetPresentedFrameNum(),
			                      "ui_target", ".rgba")) {
				captured.store(false, std::memory_order_relaxed);
			}
		}
	}
	if (ui_trace_start != UINT64_MAX && WindowGetPresentedFrameNum() >= ui_trace_start &&
	    program.shader_hash == 0xd8ad62a44c60a9a3ull && bindings.images.size() > 3) {
		static std::atomic<uint32_t> compositor_logged {0};
		if (compositor_logged.fetch_add(1, std::memory_order_relaxed) < 3) {
			const auto& binding = bindings.images[3];
			const auto& image = m_context.GetTextureCache().GetImage(binding.image_id);
			const auto  guest = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[3]);
			std::printf("UI trace: compositor image[3] guest=0x%010" PRIx64
			            " %ux%u format=%u swizzle=0x%03x"
			            " requested=0x%010" PRIx64 " %ux%u format=%u view=%u"
			            " cache_id=%u:%u cached=0x%010" PRIx64 " %ux%u format=%u"
			            " backing=%ux%u format=%u target=%u gpu_dirty=%u"
			            " buffer_dirty=%u cpu_dirty=%u\n",
			            guest.Base40(), static_cast<uint32_t>(guest.Width5()) + 1u,
			            static_cast<uint32_t>(guest.Height5()) + 1u,
			            static_cast<uint32_t>(guest.Format()), guest.DstSelXYZW(),
			            binding.desc.info.data.address, binding.desc.info.extent.width,
			            binding.desc.info.extent.height,
			            static_cast<uint32_t>(binding.desc.info.pixel_format),
			            static_cast<uint32_t>(binding.desc.view_info.format),
			            binding.image_id.index, binding.image_id.generation, image.info.data.address,
			            image.info.extent.width, image.info.extent.height,
			            static_cast<uint32_t>(image.info.pixel_format), image.backing.extent.width,
			            image.backing.extent.height, static_cast<uint32_t>(image.backing.format),
			            image.usage.render_target ? 1u : 0u, image.IsGpuModified() ? 1u : 0u,
			            image.IsBufferModified() ? 1u : 0u, image.IsCpuDirty() ? 1u : 0u);
			std::fflush(stdout);
		}
	}

	auto              vk_buffer        = buffer.Handle();
	PreparedBindings* descriptor_stage = &bindings;
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	bool has_storage_writes = HasShaderBufferWrites(input_info.stage);
	has_storage_writes =
	    std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& image) {
		                return image.written &&
		                       image.resource_class ==
		                           ShaderRecompiler::IR::ImageResourceClass::Storage;
	                }) ||
	    has_storage_writes;
	if (has_storage_writes) {
		// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
		// while allowing the queue to execute asynchronously.
		ShaderWriteHazardBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	const bool                   nan_trace = NanTraceEnabledFor(program.shader_hash);
	std::vector<uint8_t>         nan_written;
	std::vector<NanTraceCapture> nan_before;
	if (nan_trace) {
		nan_written.resize(program.info.buffers.size());
		for (size_t i = 0; i < program.info.buffers.size(); ++i) {
			nan_written[i] = program.info.buffers[i].written ? 1u : 0u;
		}
		nan_before = QueueNanTraceCopies(m_context, bindings, nan_written,
		                                 program.bindings.memory_offset_dword,
		                                 program.bindings.memory_offset_count, false);
	}
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	buffer.NoteDebugShader(program.shader_hash);
	{
		const auto label = fmt::format("Guest CS 0x{:016x} hash=0x{:016x} groups={}x{}x{}",
		                               sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		                               thread_group_x, thread_group_y, thread_group_z);
		VulkanDebugLabelScope scope(vk_buffer, label.c_str());
		vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
	}
	if (nan_trace) {
		auto nan_after = QueueNanTraceCopies(m_context, bindings, nan_written,
		                                     program.bindings.memory_offset_dword,
		                                     program.bindings.memory_offset_count, true);
		QueueNanTraceReport(m_context, std::move(nan_before), std::move(nan_after),
		                    program.shader_hash, WindowGetPresentedFrameNum(), thread_group_x,
		                    thread_group_y, thread_group_z);
	}
	// Compare the final compositor's output with its already-captured UI input on the
	// same frame. Only the opt-in diagnostic reads back one video-out image per run.
	if (dump_ui_target && program.shader_hash == 0xd8ad62a44c60a9a3ull &&
	    ui_trace_start != UINT64_MAX && WindowGetPresentedFrameNum() >= ui_trace_start) {
		static std::atomic<bool> mask_captured {false};
		if (!mask_captured.exchange(true, std::memory_order_relaxed) &&
		    !QueueUiCompositorMaskDump(m_context, bindings, WindowGetPresentedFrameNum(),
		                               program.bindings.memory_offset_dword,
		                               program.info.buffers.size() > 4
		                                   ? program.info.buffers[4].first_use_pc : 0)) {
			mask_captured.store(false, std::memory_order_relaxed);
			static std::atomic<bool> warned {false};
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				std::printf("UI trace: compositor mask readback unavailable buffers=%zu"
				            " sources=%zu\n", bindings.buffers.size(),
				            bindings.buffer_sources.size());
				std::fflush(stdout);
			}
		}
		static std::atomic<bool> flag_captured {false};
		if (!flag_captured.exchange(true, std::memory_order_relaxed) &&
		    !QueueUiCompositorParameterDump(m_context, bindings, WindowGetPresentedFrameNum(),
		                                    program.bindings.memory_offset_dword)) {
			flag_captured.store(false, std::memory_order_relaxed);
			static std::atomic<bool> warned {false};
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				const auto source_size = bindings.buffer_sources.size() > 1
				                             ? bindings.buffer_sources[1].size : 0;
				const auto bound_range = bindings.buffers.size() > 1
				                             ? bindings.buffers[1].range : 0;
				std::printf("UI trace: compositor parameter readback unavailable"
				            " sources=%zu bound=%zu data=%zu offset_dword=%u"
				            " source_size=%llu bound_range=%llu\n",
				            bindings.buffer_sources.size(), bindings.buffers.size(),
				            bindings.shader_data.size(), program.bindings.memory_offset_dword,
				            static_cast<unsigned long long>(source_size),
				            static_cast<unsigned long long>(bound_range));
				std::fflush(stdout);
			}
		}
		static std::atomic<bool> output_captured {false};
		if (!output_captured.load(std::memory_order_relaxed)) {
			bool found_output = false;
			for (size_t i = 0; i < program.info.images.size() && i < bindings.images.size(); ++i) {
				const auto& resource = program.info.images[i];
				if (!resource.written ||
				    resource.resource_class != ShaderRecompiler::IR::ImageResourceClass::Storage)
					continue;
				auto& image = m_context.GetTextureCache().GetImage(bindings.images[i].image_id);
				if (image.backing.extent.width != 1920 || image.backing.extent.height != 1080 ||
				    (image.backing.format != vk::Format::eA2B10G10R10UnormPack32 &&
				     image.backing.format != vk::Format::eA2R10G10B10UnormPack32))
					continue;
				found_output = true;
				if (!output_captured.exchange(true, std::memory_order_relaxed) &&
				    !QueueUiImageDump(m_context, image, WindowGetPresentedFrameNum(),
				                      "compositor_output", ".packed10")) {
					output_captured.store(false, std::memory_order_relaxed);
				}
				break;
			}
			if (!found_output) {
				static std::atomic<bool> warned {false};
				if (!warned.exchange(true, std::memory_order_relaxed)) {
					std::printf("UI trace: compositor output readback skipped: no 1920x1080"
					            " packed-10 storage image\n");
					std::fflush(stdout);
				}
			}
		}
	}

	// The removed host fence also ordered read-only dispatches before later writers.
	ShaderAccessBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	ResetBindings();
}

} // namespace Libs::Graphics
