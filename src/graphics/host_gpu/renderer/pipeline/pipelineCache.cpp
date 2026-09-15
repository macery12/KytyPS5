#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/perfStats.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineRecord.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "kytyGitVersion.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
	std::abort();
}

constexpr uint64_t MaxDriverCacheFileSize  = 1024ull * 1024ull * 1024ull;
constexpr uint32_t DriverCacheSaveInterval = 128;

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// Not keyed by emulator revision: the driver keys entries by shader module and create-info
	// contents, so recompiler changes only add entries. MaxDriverCacheFileSize bounds the growth.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n", properties.vendorID,
	                   properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	if (Log::GetDirection() != Log::Direction::Console) {
		std::fwrite(message.data(), 1, message.size(), stdout);
		std::fflush(stdout);
	}
	Log::Write(message);
	Log::Flush();
}

bool ReadShaderGuestMemory(void*, uint64_t address, uint32_t* value) {
	return value != nullptr &&
	       Libs::LibKernel::Memory::TryReadGpuCleanBacking(address, value, sizeof(*value));
}

// Records each word the resource specialization reads, so shader warm-up can serve it back.
bool RecordShaderGuestMemory(void* userdata, uint64_t address, uint32_t* value) {
	const bool valid = ReadShaderGuestMemory(nullptr, address, value);
	static_cast<std::vector<PipelineRecord::MemoryRead>*>(userdata)->push_back(
	    {.address = address, .value = valid ? *value : 0u, .valid = valid ? 1u : 0u});
	return valid;
}

// Serves recorded words to the resource specialization during shader warm-up.
bool ReplayShaderGuestMemory(void* userdata, uint64_t address, uint32_t* value) {
	const auto& reads = *static_cast<const std::vector<PipelineRecord::MemoryRead>*>(userdata);
	const auto  read  = std::ranges::find(reads, address, &PipelineRecord::MemoryRead::address);
	if (value == nullptr || read == reads.end() || read->valid == 0) {
		return false;
	}
	*value = read->value;
	return true;
}

// Runs job(0) .. job(count - 1) on up to 32 hardware threads, including the calling thread.
void ParallelFor(size_t count, const std::function<void(size_t)>& job) {
	const size_t workers =
	    std::min<size_t>(std::clamp(std::thread::hardware_concurrency(), 1u, 32u), count);
	std::atomic<size_t> next {0};
	const auto run = [&] {
		for (size_t index = next.fetch_add(1); index < count; index = next.fetch_add(1)) {
			job(index);
		}
	};
	std::vector<std::thread> threads;
	for (size_t worker = 1; worker < workers; worker++) {
		threads.emplace_back(run);
	}
	run();
	for (auto& thread: threads) {
		thread.join();
	}
}

double SecondsSince(std::chrono::steady_clock::time_point start) {
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Writing the shaders out is what makes a recompiler problem analysable offline, and requiring
// the graphics debug dump for it is too blunt: that costs hundreds of megabytes of log and slows
// startup enough to change what reproduces. The shader log folder already exists for exactly this
// output, so File direction is enough on its own.
bool ShaderDumpEnabled() {
	return Config::GraphicsDebugDumpEnabled() ||
	       Config::GetShaderLogDirection() == Config::LogDirection::File;
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!ShaderDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code, const std::string& decoded_dump) {
	if (!ShaderDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto base = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(base.parent_path());
	for (const auto& [suffix, data, size]: {
	         std::tuple {".bin", static_cast<const void*>(code.data()), code.size_bytes()},
	         std::tuple {".rdna2", static_cast<const void*>(decoded_dump.data()),
	                     decoded_dump.size()},
	     }) {
		if (size == 0) {
			continue;
		}
		auto path = base;
		path += suffix;
		Common::File file(path);
		if (file.IsInvalid()) {
			const auto path_text = Common::PathToString(path);
			LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		} else {
			file.Write(data, size);
		}
	}
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	if (tools.Validate(spirv)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		explicit SourceEntry(ShaderRecompiler::IR::ResourcePlan plan)
		    : resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		ShaderRecompiler::IR::ResourcePlan resource_plan;
		std::vector<Permutation>           permutations;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing up to 429 words first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 13 + ShaderVertexInputInfo::RES_MAX * 13;

	Permutation CompilePermutation(const ShaderParams&                          params,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword) {
		KYTY_PROFILER_FUNCTION();
		const char* stage_name = nullptr;
		switch (options.stage) {
			case ShaderType::Vertex: stage_name = "vs"; break;
			case ShaderType::Mesh: stage_name = "ms"; break;
			case ShaderType::Local: stage_name = "ls"; break;
			case ShaderType::TessellationControl: stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: stage_name = "ds"; break;
			case ShaderType::Pixel: stage_name = "ps"; break;
			case ShaderType::Compute: stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		DumpShaderOriginal(stage_name, options.shader_hash, params.code, result.decoded_dump);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, device);
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = ++next_shader_id, .module = module},
		};
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = static_cast<uint32_t>(params.user_data.size());
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto                                         entry = programs.find(lookup_key);
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		const ShaderRecompiler::IR::SrtRuntime       runtime {
		    .user_data                  = params.user_data,
		    .shader_base                = params.Base(),
		    .read_specialization_memory = ReadShaderGuestMemory,
		};
		if (entry != programs.end()) {
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    entry->second.resource_plan, runtime, resources, specialization));
			if (const auto permutation = std::ranges::find_if(
			        entry->second.permutations, [&](const Permutation& candidate) {
				        const auto& layout = candidate.program.bindings;
				        return layout.push_data_start_dword ==
				                   ShaderRecompiler::IR::PushData::StartFor(
				                       push_data_cursor, layout.ShaderDataDwords()) &&
				               candidate.specialization == specialization;
			        });
			    permutation != entry->second.permutations.end()) {
				input_info.stage = {.program   = &permutation->program,
				                    .resources = std::move(resources)};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		PerfStats::Add(PerfStats::Counter::ShaderCompiles);
		PerfStats::ScopedDuration compile_time(PerfStats::Duration::ShaderCompile);
		const uint32_t compile_push_data_cursor = push_data_cursor;
		KYTY_PROFILER_BLOCK("PipelineCache::CompileShader", profiler::colors::Amber300);
		const auto options = MakeOptions(params, input_info, stage);
		KYTY_PROFILER_BLOCK("ShaderRecompiler::TranslateProgram");
		auto translated = ShaderRecompiler::TranslateProgram(params.code, options);
		KYTY_PROFILER_END_BLOCK;
		if (entry == programs.end()) {
			auto resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(resource_plan, runtime, resources,
			                                                    specialization));
			entry = programs.try_emplace(lookup_key, std::move(resource_plan)).first;
		}
		entry->second.permutations.push_back(CompilePermutation(
		    params, options, std::move(translated), std::move(specialization), push_data_cursor));
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = std::move(resources)};
		permutation.program.bindings.AdvancePushData(push_data_cursor);
		RecordProgram(params, input_info, stage, compile_push_data_cursor,
		              entry->second.resource_plan, permutation);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	template <typename InputInfo>
	static ShaderRecompiler::CompileOptions MakeOptions(const ShaderParams& params,
	                                                    const InputInfo& input_info, ShaderType stage) {
		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.shader_hash = params.hash;
		options.user_data   = params.user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		return options;
	}

	template <typename InputInfo>
	static ShaderType StageOf(const InputInfo& input_info) {
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			return input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			return ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			return ShaderType::Compute;
		}
	}

	template <typename InputInfo>
	void RecordProgram(const ShaderParams& params, const InputInfo& input_info, ShaderType stage,
	                   uint32_t push_data_cursor, const ShaderRecompiler::IR::ResourcePlan& plan,
	                   const Permutation& permutation) {
		if (record == nullptr || !record->IsOpen() || !PipelineRecord::IsRecordableStage(stage)) {
			return;
		}
		PipelineRecord::Program program;
		program.stage            = stage;
		program.hash             = params.hash;
		program.shader_base      = params.Base();
		program.push_data_cursor = push_data_cursor;
		program.code.assign(params.code.begin(), params.code.end());
		program.back_code.assign(params.back_code.begin(), params.back_code.end());
		program.user_data = params.user_data;
		// Derive the specialization again through a recording reader, so replay reads the same words.
		const ShaderRecompiler::IR::SrtRuntime runtime {
			.user_data                  = params.user_data,
			.shader_base                = params.Base(),
			.userdata                   = &program.reads,
			.read_specialization_memory = RecordShaderGuestMemory,
		};
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		if (!ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization) ||
		    !(specialization == permutation.specialization)) {
			return;
		}
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			program.vertex = input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			program.pixel = input_info;
		} else {
			program.compute = input_info;
		}
		if (const auto ordinal = record->AppendProgram(program); ordinal != PipelineRecord::NoProgram) {
			ordinals.try_emplace(permutation.handle.id, ordinal);
		}
	}

	struct Prepared {
		ProgramKey                         key;
		ShaderRecompiler::IR::ResourcePlan plan;
		Permutation                        permutation;
	};

	// Translates and emits one recorded program. Runs on warm-up worker threads, so it touches only its
	// own state. Returns nullopt when the recorded reads no longer derive a specialization; the game then
	// compiles and records the program again.
	std::optional<Prepared> Prepare(const PipelineRecord::Program& recorded) {
		const ShaderParams params {
			.code      = recorded.code,
			.user_data = recorded.user_data,
			.hash      = recorded.hash,
			.back_code = recorded.back_code,
			.base      = recorded.shader_base,
		};
		switch (recorded.stage) {
			case ShaderType::Vertex: return Prepare(params, recorded.vertex, recorded);
			case ShaderType::Pixel: return Prepare(params, recorded.pixel, recorded);
			case ShaderType::Compute: return Prepare(params, recorded.compute, recorded);
			default: return std::nullopt;
		}
	}

	template <typename InputInfo>
	std::optional<Prepared> Prepare(const ShaderParams& params, const InputInfo& input_info,
	                                const PipelineRecord::Program& recorded) {
		const auto stage = StageOf(input_info);
		ProgramKey key {
			.stage           = stage,
			.hash            = params.hash,
			.user_data_count = static_cast<uint32_t>(params.user_data.size()),
			.code_size       = static_cast<uint32_t>(params.code.size()),
		};
		BuildStageStaticKey(input_info, key.static_state);
		const auto options    = MakeOptions(params, input_info, stage);
		auto       translated = ShaderRecompiler::TranslateProgram(params.code, options);
		auto       plan       = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		const ShaderRecompiler::IR::SrtRuntime runtime {
			.user_data                  = params.user_data,
			.shader_base                = params.Base(),
			.userdata                   = const_cast<void*>(static_cast<const void*>(&recorded.reads)),
			.read_specialization_memory = ReplayShaderGuestMemory,
		};
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		if (!ShaderRecompiler::IR::MaterializeResources(plan, runtime, resources, specialization)) {
			return std::nullopt;
		}
		auto permutation = CompilePermutation(params, options, std::move(translated),
		                                      std::move(specialization), recorded.push_data_cursor);
		return Prepared {
			.key         = std::move(key),
			.plan        = std::move(plan),
			.permutation = std::move(permutation),
		};
	}

	// Adds a warmed-up program unless the same permutation is already cached.
	ShaderProgram Insert(Prepared prepared, uint32_t ordinal) {
		auto entry = programs.find(prepared.key);
		if (entry == programs.end()) {
			entry = programs.try_emplace(std::move(prepared.key), std::move(prepared.plan)).first;
		}
		auto&      permutations = entry->second.permutations;
		const auto existing = std::ranges::find_if(permutations, [&](const Permutation& candidate) {
			return candidate.program.bindings.push_data_start_dword ==
			           prepared.permutation.program.bindings.push_data_start_dword &&
			       candidate.specialization == prepared.permutation.specialization;
		});
		ShaderProgram handle;
		if (existing != permutations.end()) {
			device.destroyShaderModule(prepared.permutation.handle.module, nullptr);
			handle = existing->handle;
		} else {
			permutations.push_back(std::move(prepared.permutation));
			handle = permutations.back().handle;
		}
		ordinals.try_emplace(handle.id, ordinal);
		return handle;
	}

	[[nodiscard]] uint32_t Ordinal(uint64_t shader_id) const {
		const auto found = ordinals.find(shader_id);
		return found != ordinals.end() ? found->second : PipelineRecord::NoProgram;
	}

	[[nodiscard]] std::unordered_map<uint64_t, const ShaderRecompiler::IR::CompiledShaderInfo*>
	CompiledPrograms() const {
		std::unordered_map<uint64_t, const ShaderRecompiler::IR::CompiledShaderInfo*> compiled;
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				compiled.emplace(permutation.handle.id, &permutation.program);
			}
		}
		return compiled;
	}

	explicit ProgramCache(vk::Device device): device(device) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	vk::Device                                                  device;
	std::atomic<uint64_t>                                       next_shader_id {0};
	PipelineRecord::RecordFile*                                 record = nullptr;
	// Shader handle id -> ordinal of the Program record it was compiled from.
	std::unordered_map<uint64_t, uint32_t>                      ordinals;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics.device)) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
	InitializeShaderRecord();
}

PipelineCache::~PipelineCache() {
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}

	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= MaxDriverCacheFileSize) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, emulator, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {} bytes from {}", initial_data.size(),
		                 path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::Save() {
	Common::LockGuard lock(m_mutex);
	if (m_driver_cache == nullptr) {
		return;
	}
	WriteDriverCache();
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	m_driver_cache = nullptr;
}

void PipelineCache::NoteNewPipeline() {
	PerfStats::Add(PerfStats::Counter::PipelineCompiles);
	if (m_driver_cache == nullptr || ++m_pipelines_since_save < DriverCacheSaveInterval) {
		return;
	}
	m_pipelines_since_save = 0;
	WriteDriverCache();
}

void PipelineCache::WriteDriverCache() {
	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {} bytes)",
		                 vk::to_string(result), size);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	if (prefix_written != prefix.size() || payload_written != payload.size() || !flushed ||
	    !Common::File::RenameFile(temp_path, m_driver_cache_path)) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {} bytes to {}", payload.size(),
	                 Common::PathToString(m_driver_cache_path));
}

void PipelineCache::InitializeShaderRecord() {
	if (!Config::PreGenEnabled()) {
		PipelineCacheLog("Shader warm-up: disabled (--pre-gen false)");
		return;
	}
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	const auto path = std::filesystem::path("_PipelineCache") / (title_id + ".shaders");
	PipelineRecord::Records records;
	m_record = std::make_unique<PipelineRecord::RecordFile>();
	if (!m_record->Open(path, records)) {
		PipelineCacheLog("Shader warm-up: cannot write {}; new shaders are not recorded",
		                 Common::PathToString(path));
	}
	m_program_cache->record = m_record.get();
	WarmUp(records);
}

void PipelineCache::WarmUp(PipelineRecord::Records& records) {
	if (records.programs.empty()) {
		PipelineCacheLog(
		    "Shader warm-up: nothing recorded yet; shaders are recorded as the game compiles them");
		return;
	}
	KYTY_PROFILER_FUNCTION();
	const auto start = std::chrono::steady_clock::now();
	PipelineCacheLog("Shader warm-up: compiling {} programs, {} graphics and {} compute pipelines",
	                 records.programs.size(), records.graphics.size(), records.compute.size());

	std::vector<std::optional<ProgramCache::Prepared>> prepared(records.programs.size());
	ParallelFor(prepared.size(), [&](size_t index) {
		if (auto program = m_program_cache->Prepare(records.programs[index]); program.has_value()) {
			prepared[index].emplace(std::move(*program));
		}
	});

	Common::LockGuard          lock(m_mutex);
	std::vector<ShaderProgram> handles(records.programs.size());
	size_t                     skipped_programs = 0;
	for (size_t index = 0; index < prepared.size(); index++) {
		if (!prepared[index].has_value()) {
			skipped_programs++;
			continue;
		}
		handles[index] =
		    m_program_cache->Insert(std::move(*prepared[index]), static_cast<uint32_t>(index));
	}
	prepared.clear();
	const auto compiled        = m_program_cache->CompiledPrograms();
	const auto program_seconds = SecondsSince(start);

	struct GraphicsJob {
		const PipelineRecord::GraphicsPipeline* record = nullptr;
		GraphicsPipelineKey                     key;
		GraphicsPrograms                        programs;
		std::unique_ptr<Pipeline>               pipeline;
	};
	std::vector<GraphicsJob>                                         graphics_jobs;
	std::unordered_set<GraphicsPipelineKey, GraphicsPipelineKeyHash> graphics_keys;
	for (const auto& record: records.graphics) {
		const auto vertex    = handles[record.vertex_program];
		const bool ps_active = record.pixel_program != PipelineRecord::NoProgram;
		const auto pixel     = ps_active ? handles[record.pixel_program] : ShaderProgram {};
		if (!vertex || (ps_active && !pixel)) {
			continue;
		}
		GraphicsJob job {.record = &record};
		job.key.rendering            = record.rendering;
		job.key.vertex_shader_ids[0] = vertex.id;
		job.key.ps_shader_id         = pixel.id;
		job.key.vertex_input         = record.vertex_input;
		job.key.static_params        = record.static_params;
		job.programs.vertex[0]       = vertex;
		job.programs.pixel           = pixel;
		if (m_graphics_pipelines.contains(job.key) || !graphics_keys.insert(job.key).second) {
			continue;
		}
		graphics_jobs.push_back(std::move(job));
	}
	ParallelFor(graphics_jobs.size(), [&](size_t index) {
		auto& job                 = graphics_jobs[index];
		auto  vertex_info         = job.record->vertex;
		vertex_info.stage.program = compiled.at(job.programs.vertex[0].id);
		auto       pixel_info     = job.record->pixel;
		const bool ps_active      = static_cast<bool>(job.programs.pixel);
		if (ps_active) {
			pixel_info.stage.program = compiled.at(job.programs.pixel.id);
		}
		job.pipeline = std::make_unique<Pipeline>();
		CreatePipelineInternal(m_graphics, *job.pipeline, job.key.rendering, job.key.vertex_input,
		                       std::span<const ShaderVertexInputInfo>(&vertex_info, 1),
		                       ps_active ? &pixel_info : nullptr, job.programs,
		                       job.key.static_params, m_driver_cache);
	});
	for (auto& job: graphics_jobs) {
		EXIT_NOT_IMPLEMENTED(job.pipeline->pipeline == nullptr ||
		                     job.pipeline->pipeline_layout == nullptr);
		m_graphics_pipelines.emplace(job.key, std::move(job.pipeline));
	}

	struct ComputeJob {
		const PipelineRecord::ComputePipeline* record = nullptr;
		ShaderProgram                          program;
		std::unique_ptr<Pipeline>              pipeline;
	};
	std::vector<ComputeJob>      compute_jobs;
	std::unordered_set<uint64_t> compute_ids;
	for (const auto& record: records.compute) {
		const auto program = handles[record.program];
		if (!program || m_compute_pipelines.contains(program.id) ||
		    !compute_ids.insert(program.id).second) {
			continue;
		}
		compute_jobs.push_back({.record = &record, .program = program});
	}
	ParallelFor(compute_jobs.size(), [&](size_t index) {
		auto& job          = compute_jobs[index];
		auto  info         = job.record->compute;
		info.stage.program = compiled.at(job.program.id);
		job.pipeline       = std::make_unique<Pipeline>();
		CreatePipelineInternal(m_graphics, *job.pipeline, info, job.program.module, m_driver_cache);
	});
	for (auto& job: compute_jobs) {
		EXIT_NOT_IMPLEMENTED(job.pipeline->pipeline == nullptr ||
		                     job.pipeline->pipeline_layout == nullptr);
		m_compute_pipelines.emplace(job.program.id, std::move(job.pipeline));
	}

	if (m_driver_cache != nullptr) {
		WriteDriverCache();
	}
	PipelineCacheLog("Shader warm-up: {} programs ({} skipped) in {:.1f}s, {} graphics and {} "
	                 "compute pipelines in {:.1f}s",
	                 records.programs.size() - skipped_programs, skipped_programs, program_seconds,
	                 graphics_jobs.size(), compute_jobs.size(), SecondsSince(start) - program_seconds);
}

void PipelineCache::RecordGraphicsPipeline(const GraphicsPipelineKey&             key,
                                           std::span<const ShaderVertexInputInfo> vertex_info,
                                           const ShaderPixelInputInfo*            ps_input_info) {
	if (m_record == nullptr || !m_record->IsOpen() || vertex_info.size() != 1 ||
	    vertex_info[0].logical_stage != ShaderType::Vertex) {
		return;
	}
	PipelineRecord::GraphicsPipeline record;
	record.vertex_program = m_program_cache->Ordinal(key.vertex_shader_ids[0]);
	record.pixel_program  = ps_input_info != nullptr ? m_program_cache->Ordinal(key.ps_shader_id)
	                                                 : PipelineRecord::NoProgram;
	if (record.vertex_program == PipelineRecord::NoProgram ||
	    (ps_input_info != nullptr && record.pixel_program == PipelineRecord::NoProgram)) {
		return;
	}
	record.rendering     = key.rendering;
	record.vertex_input  = key.vertex_input;
	record.static_params = key.static_params;
	record.vertex        = vertex_info[0];
	if (ps_input_info != nullptr) {
		record.pixel = *ps_input_info;
	}
	m_record->AppendGraphicsPipeline(record);
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    bool dual_source_blend, std::array<ShaderVertexInputInfo, 3>& vertex_info,
    ShaderPixelInputInfo& pixel_info) {
	KYTY_PROFILER_FUNCTION();
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		// PrepareProgram rebuilds pixel_info from the guest registers, so anything derived
		// from render state has to be applied after it.
		pixel_info.ps_dual_source_blend = dual_source_blend;
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	Common::LockGuard lock(m_mutex);
	uint32_t          push_data_cursor =
	    mesh_active ? ShaderRecompiler::IR::PushData::MeshDrawDwordCount : 0;
	GraphicsPrograms  result;
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
	}
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	Common::LockGuard lock(m_mutex);
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	Common::LockGuard lock(m_mutex);
	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = color_count;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		auto target_mask = render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot);
		// A slot left out of CB_SHADER_MASK receives no pixel-shader export (exports are packed onto
		// the enabled slots), so the hardware leaves it untouched. Vulkan leaves an attachment with
		// no output undefined; mask it off instead.
		const auto shader_mask = ctx.GetShaderRegisters().m_cbShaderMask;
		if (ps_active && shader_mask != 0 &&
		    render_target_mask_slot(shader_mask, colors[i].target_slot) == 0) {
			target_mask = 0;
		}
		static_params.color_mask[i] = colors[i].export_mapping.ApplyMask(target_mask);
		rendering.color_formats[i] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		static_params.color_srcblend[i]       = bc.color_srcblend;
		static_params.color_comb_fcn[i]       = bc.color_comb_fcn;
		static_params.color_destblend[i]      = bc.color_destblend;
		static_params.alpha_srcblend[i]       = bc.alpha_srcblend;
		static_params.alpha_comb_fcn[i]       = bc.alpha_comb_fcn;
		static_params.alpha_destblend[i]      = bc.alpha_destblend;
		static_params.separate_alpha_blend[i] = bc.separate_alpha_blend;
		static_params.blend_enable[i]         = bc.enable && !rt.info.blend_bypass;
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	static_params.depth_bounds_test_enable = depth.depth_bounds_test_enable;
	static_params.depth_min_bounds         = depth.depth_min_bounds;
	static_params.depth_max_bounds         = depth.depth_max_bounds;
	const bool rect_list =
	    command.GetUserConfig().GetPrimType() == Prospero::PrimitiveType::kRectList;
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		uint32_t attributes_num          = 0;
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			EXIT_IF(buffer.attr_num < 0 || buffer.attr_num > ShaderVertexInputBuffer::ATTR_MAX);
			attributes_num += static_cast<uint32_t>(buffer.attr_num);
			EXIT_IF(attributes_num > static_cast<uint32_t>(vs_input_info.resources_num));
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
			for (int attribute = 0; attribute < buffer.attr_num; attribute++) {
				const auto index = buffer.attr_indices[attribute];
				EXIT_IF(index < 0 || index >= vs_input_info.resources_num);
				key.vertex_input.attributes[index] = {
				    .offset  = buffer.attr_offsets[attribute],
				    .binding = static_cast<uint8_t>(binding),
				};
			}
		}
		EXIT_IF(attributes_num != static_cast<uint32_t>(vs_input_info.resources_num));
	}

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	PerfStats::ScopedDuration compile_time(PerfStats::Duration::PipelineCompile);
	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
	                       ps_input_info, programs, static_params, m_driver_cache);
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	NoteNewPipeline();
	RecordGraphicsPipeline(iter->first, vertex_info, ps_input_info);

	return *iter->second;
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	Common::LockGuard lock(m_mutex);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	PerfStats::ScopedDuration compile_time(PerfStats::Duration::PipelineCompile);
	auto cached = std::make_unique<Pipeline>();
	CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module, m_driver_cache);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	NoteNewPipeline();
	if (const auto ordinal = m_program_cache->Ordinal(compute_program.id);
	    m_record != nullptr && ordinal != PipelineRecord::NoProgram) {
		PipelineRecord::ComputePipeline record;
		record.program = ordinal;
		record.compute = input_info;
		m_record->AppendComputePipeline(record);
	}

	return *iter->second;
}
} // namespace Libs::Graphics
