#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANCOMMON_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANCOMMON_H_

#define VK_NO_PROTOTYPES
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC    1
#define VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL 1
#define VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_EXCEPTIONS

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#include "common/emulatorConfig.h"
#include "graphics/guest_gpu/gpu_defs.h"

#include <cstdint>
#include <cstdlib>
#include <fmt/format.h>
#include <span>
#include <string>
#include <utility>
#include <vector>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_hash.hpp>

namespace Libs::Graphics {

using VulkanMemoryBarrier = vk::MemoryBarrier;

vk::Format  VulkanFormat(Prospero::BufferFormat guest_format);
void        RequireVulkanSuccess(vk::Result result, const char* operation);
vk::ShaderModule CompileSPV(std::span<const uint32_t> code, vk::Device device);

// Guest draws and dispatches are named in the command stream for GPU crash tools (Radeon GPU
// Detective, RenderDoc) only on request: formatting a label for every draw costs measurable CPU
// in busy scenes. KYTY_GPU_LABELS=1 or the graphics debug dump enables them.
[[nodiscard]] inline bool GpuDebugLabelsEnabled() noexcept {
	static const bool requested = [] {
		const char* value = std::getenv("KYTY_GPU_LABELS");
		return (value != nullptr && value[0] == '1' && value[1] == '\0') ||
		       Config::GraphicsDebugDumpEnabled();
	}();
	return requested && VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBeginDebugUtilsLabelEXT != nullptr;
}

class VulkanDebugLabelScope {
public:
	VulkanDebugLabelScope(vk::CommandBuffer command, const char* name): m_command(command) {
		if (command == nullptr || name == nullptr ||
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBeginDebugUtilsLabelEXT == nullptr ||
		    VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdEndDebugUtilsLabelEXT == nullptr) {
			return;
		}
		vk::DebugUtilsLabelEXT label {};
		label.pLabelName = name;
		command.beginDebugUtilsLabelEXT(&label);
		m_active = true;
	}

	~VulkanDebugLabelScope() {
		if (m_active) {
			m_command.endDebugUtilsLabelEXT();
		}
	}

	VulkanDebugLabelScope(const VulkanDebugLabelScope&)            = delete;
	VulkanDebugLabelScope& operator=(const VulkanDebugLabelScope&) = delete;

private:
	vk::CommandBuffer m_command = nullptr;
	bool              m_active  = false;
};

template <typename Handle, typename... Args>
void SetVulkanObjectNameF(vk::Device device, Handle handle, fmt::format_string<Args...> format,
                          Args&&... args) {
	if (!Config::GraphicsDebugDumpEnabled() || device == nullptr || handle == nullptr ||
	    VULKAN_HPP_DEFAULT_DISPATCHER.vkSetDebugUtilsObjectNameEXT == nullptr) {
		return;
	}

	const auto                      name = fmt::format(format, std::forward<Args>(args)...);
	vk::DebugUtilsObjectNameInfoEXT info {};
	info.objectType   = Handle::objectType;
	info.objectHandle = static_cast<uint64_t>(
	    reinterpret_cast<uintptr_t>(static_cast<typename Handle::CType>(handle)));
	info.pObjectName = name.c_str();
	(void)device.setDebugUtilsObjectNameEXT(&info);
}

template <typename T, typename Enumerator>
[[nodiscard]] std::vector<T> EnumerateVulkan(const char* operation, Enumerator&& enumerate) {
	for (;;) {
		uint32_t count = 0;
		RequireVulkanSuccess(enumerate(&count, nullptr), operation);
		if (count == 0) {
			return {};
		}

		std::vector<T> values(count);
		const auto     result = enumerate(&count, values.data());
		if (result == vk::Result::eSuccess) {
			values.resize(count);
			return values;
		}
		if (result != vk::Result::eIncomplete) {
			RequireVulkanSuccess(result, operation);
		}
	}
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_VULKANCOMMON_H_
