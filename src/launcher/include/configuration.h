#ifndef LAUNCHER_INCLUDE_CONFIGURATION_H_
#define LAUNCHER_INCLUDE_CONFIGURATION_H_

#include "common.h"
#include "common/emulatorConfig.h"

#include <QByteArray>
#include <QChar>
#include <QList>
#include <QMetaEnum>
#include <QMetaType>
#include <QSettings>
#include <QString>
#include <QStringList>
#include <QVariant>

#define KYTY_CFG_SET(n) s->setValue(#n, QVariant::fromValue(n).toString());
#define KYTY_CFG_GET(n) n = s->value(#n).value<decltype(n)>();
// Keeps the member's default when the key is missing, e.g. in settings saved by an older launcher.
#define KYTY_CFG_GET_OR(n) n = s->value(#n, QVariant::fromValue(n)).value<decltype(n)>();

template <class T>
inline QStringList EnumToList() {
	QStringList ret;
	auto        me    = QMetaEnum::fromType<T>();
	int         count = me.keyCount();
	for (int i = 0; i < count; i++) {
		auto key = QString(me.key(i));
		ret << (key.startsWith('R') && key.size() > 2 && key.at(1).isDigit()
		            ? key.remove('R').toLower()
		            : key);
	}
	return ret;
}

template <class T>
T TextToEnum(const QString& text) {
	auto me = QMetaEnum::fromType<T>();
	return static_cast<T>(me.keyToValue(
	    ((text.size() > 1 && text.at(0).isDigit()) ? 'R' + text.toUpper() : text).toUtf8().data()));
}

template <class T>
QString EnumToText(T value) {
	auto me  = QMetaEnum::fromType<T>();
	auto key = QString(me.valueToKey(static_cast<int>(value)));
	return (key.startsWith('R') && key.size() > 2 && key.at(1).isDigit() ? key.remove('R').toLower()
	                                                                     : key);
}

class Configuration: public QObject {
	Q_OBJECT

public:
	static constexpr int DEFAULT_CONSOLE_LANGUAGE = 1;
	static constexpr int MAX_CONSOLE_LANGUAGE     = 29;

	enum class Resolution {
		R1280X720,
		R1920X1080,
		R2560X1440,
		R3840X2160,
	};
	Q_ENUM(Resolution)

	enum class ShaderOptimizationType { None, Size, Performance };
	Q_ENUM(ShaderOptimizationType)

	enum class PresentMode { Fifo, Mailbox, Immediate };
	Q_ENUM(PresentMode)

	enum class LogDirection { Silent, Console, File };
	Q_ENUM(LogDirection)

	enum class GameStatus { Unknown, InGame, Logo, DoesntBoot, MainMenu };
	Q_ENUM(GameStatus)

	// One variable of the emulator's environment. `set == false` removes the variable, so a value
	// inherited from the launcher's own environment cannot override the setting.
	struct EnvironmentVariable {
		QString name;
		QString value;
		bool    set = true;
	};

	Configuration() = default;

	QString    name;
	QString    title_id;    /* Serial / title id from sce_sys/param.json */
	QString    gameVersion; /* appVersion / contentVersion from sce_sys/param.json */
	QString    firmwareVer; /* requiredSystemSoftwareVersion from sce_sys/param.json */
	QString    basedir;     /* Game base directory */
	QString    game_path;   /* Launcher-unique game path */
	bool       custom_settings = false;
	GameStatus game_status     = GameStatus::Unknown;
	QString    game_comment;

	Resolution             screen_resolution           = Resolution::R1280X720;
	QString                user_name                   = "Kyty";
	int                    user_id                     = Config::DEFAULT_USER_ID;
	PresentMode            present_mode                = PresentMode::Mailbox;
	int                    gpu_index                   = -1;
	bool                   fullscreen_enabled          = false;
	bool                   readback_linear_images      = false;
	int                    vblank_frequency            = 60;
	int                    console_language            = DEFAULT_CONSOLE_LANGUAGE;
	bool                   vulkan_validation_enabled   = false;
	bool                   shader_validation_enabled   = true;
	ShaderOptimizationType shader_optimization_type    = ShaderOptimizationType::Performance;
	LogDirection           shader_log_direction        = LogDirection::Silent;
	QString                shader_log_folder           = "_Shaders";
	bool                   command_buffer_dump_enabled = false;
	QString                command_buffer_dump_folder  = "_Buffers";
	LogDirection           printf_direction            = LogDirection::Silent;
	QString                printf_output_file          = "_kyty.txt";
	bool                   profiler_enabled            = false;
	bool                   renderdoc_enabled           = false;
#if defined(_WIN32)
	bool red_zone_protection_enabled = false;
#endif
	QStringList host_input_mapping;

	// Performance and diagnostics passed as command-line options.
	bool pre_gen_enabled                 = true;
	bool gpu_assisted_validation_enabled = false;
	bool graphics_debug_dump_enabled     = false;
	bool spirv_debug_printf_enabled      = false;

	// Performance and diagnostics passed as KYTY_* environment variables (EmulatorEnvironment()).
	bool    perf_stats_enabled             = false;
	bool    gpu_labels_enabled             = false;
	bool    sync_compute_enabled           = false;
	bool    tessellation_enabled           = false;
	bool    force_ui_mask_enabled          = true;
	bool    pixel_quad_derivatives_enabled = true;
	bool    trace_dcc_enabled              = false;
	bool    cmask_clear_disabled           = false;
	QString shader_loop_guard_hashes;
	QString shader_loop_limit;
	QString skip_cs_hashes;
	QString skip_cs_addresses;
	QString trace_nan_cs;
	// Any other variables, as NAME=VALUE entries separated by ';'.
	QString extra_environment;

	QString elf = QStringLiteral("eboot.bin");

	void CopyEmulatorSettingsFrom(const Configuration& other) {
		screen_resolution           = other.screen_resolution;
		user_name                   = other.user_name;
		user_id                     = other.user_id;
		present_mode                = other.present_mode;
		gpu_index                   = other.gpu_index;
		fullscreen_enabled          = other.fullscreen_enabled;
		readback_linear_images      = other.readback_linear_images;
		vblank_frequency            = other.vblank_frequency;
		console_language            = other.console_language;
		vulkan_validation_enabled   = other.vulkan_validation_enabled;
		shader_validation_enabled   = other.shader_validation_enabled;
		shader_optimization_type    = other.shader_optimization_type;
		shader_log_direction        = other.shader_log_direction;
		shader_log_folder           = other.shader_log_folder;
		command_buffer_dump_enabled = other.command_buffer_dump_enabled;
		command_buffer_dump_folder  = other.command_buffer_dump_folder;
		printf_direction            = other.printf_direction;
		printf_output_file          = other.printf_output_file;
		profiler_enabled            = other.profiler_enabled;
		renderdoc_enabled           = other.renderdoc_enabled;
#if defined(_WIN32)
		red_zone_protection_enabled = other.red_zone_protection_enabled;
#endif
		host_input_mapping = other.host_input_mapping;

		pre_gen_enabled                 = other.pre_gen_enabled;
		gpu_assisted_validation_enabled = other.gpu_assisted_validation_enabled;
		graphics_debug_dump_enabled     = other.graphics_debug_dump_enabled;
		spirv_debug_printf_enabled      = other.spirv_debug_printf_enabled;
		perf_stats_enabled              = other.perf_stats_enabled;
		gpu_labels_enabled              = other.gpu_labels_enabled;
		sync_compute_enabled            = other.sync_compute_enabled;
		tessellation_enabled            = other.tessellation_enabled;
		force_ui_mask_enabled           = other.force_ui_mask_enabled;
		pixel_quad_derivatives_enabled  = other.pixel_quad_derivatives_enabled;
		trace_dcc_enabled               = other.trace_dcc_enabled;
		cmask_clear_disabled            = other.cmask_clear_disabled;
		shader_loop_guard_hashes        = other.shader_loop_guard_hashes;
		shader_loop_limit               = other.shader_loop_limit;
		skip_cs_hashes                  = other.skip_cs_hashes;
		skip_cs_addresses               = other.skip_cs_addresses;
		trace_nan_cs                    = other.trace_nan_cs;
		extra_environment               = other.extra_environment;
	}

	void CopyFrom(const Configuration& other) {
		name            = other.name;
		title_id        = other.title_id;
		gameVersion     = other.gameVersion;
		firmwareVer     = other.firmwareVer;
		basedir         = other.basedir;
		game_path       = other.game_path;
		custom_settings = other.custom_settings;
		game_status     = other.game_status;
		game_comment    = other.game_comment;
		CopyEmulatorSettingsFrom(other);
		elf = other.elf;
	}

	void WriteSettings(QSettings* s) const {
		KYTY_CFG_SET(name);
		KYTY_CFG_SET(basedir);
		KYTY_CFG_SET(game_path);
		KYTY_CFG_SET(custom_settings);
		KYTY_CFG_SET(screen_resolution);
		KYTY_CFG_SET(user_name);
		KYTY_CFG_SET(user_id);
		KYTY_CFG_SET(present_mode);
		KYTY_CFG_SET(gpu_index);
		KYTY_CFG_SET(fullscreen_enabled);
		KYTY_CFG_SET(readback_linear_images);
		KYTY_CFG_SET(vblank_frequency);
		KYTY_CFG_SET(console_language);
		KYTY_CFG_SET(vulkan_validation_enabled);
		KYTY_CFG_SET(shader_validation_enabled);
		KYTY_CFG_SET(shader_optimization_type);
		KYTY_CFG_SET(shader_log_direction);
		KYTY_CFG_SET(shader_log_folder);
		KYTY_CFG_SET(command_buffer_dump_enabled);
		KYTY_CFG_SET(command_buffer_dump_folder);
		KYTY_CFG_SET(printf_direction);
		KYTY_CFG_SET(printf_output_file);
		KYTY_CFG_SET(profiler_enabled);
		KYTY_CFG_SET(renderdoc_enabled);
#if defined(_WIN32)
		KYTY_CFG_SET(red_zone_protection_enabled);
#endif
		s->setValue("host_input_mapping", host_input_mapping);
		KYTY_CFG_SET(pre_gen_enabled);
		KYTY_CFG_SET(gpu_assisted_validation_enabled);
		KYTY_CFG_SET(graphics_debug_dump_enabled);
		KYTY_CFG_SET(spirv_debug_printf_enabled);
		KYTY_CFG_SET(perf_stats_enabled);
		KYTY_CFG_SET(gpu_labels_enabled);
		KYTY_CFG_SET(sync_compute_enabled);
		KYTY_CFG_SET(tessellation_enabled);
		KYTY_CFG_SET(force_ui_mask_enabled);
		KYTY_CFG_SET(pixel_quad_derivatives_enabled);
		KYTY_CFG_SET(trace_dcc_enabled);
		KYTY_CFG_SET(cmask_clear_disabled);
		KYTY_CFG_SET(shader_loop_guard_hashes);
		KYTY_CFG_SET(shader_loop_limit);
		KYTY_CFG_SET(skip_cs_hashes);
		KYTY_CFG_SET(skip_cs_addresses);
		KYTY_CFG_SET(trace_nan_cs);
		KYTY_CFG_SET(extra_environment);
		KYTY_CFG_SET(elf);
	}

	void ReadSettings(QSettings* s) {
		KYTY_CFG_GET(name);
		KYTY_CFG_GET(basedir);
		KYTY_CFG_GET(game_path);
		KYTY_CFG_GET(custom_settings);
		KYTY_CFG_GET(screen_resolution);
		user_name          = s->value("user_name", user_name).toString();
		bool user_id_ok    = false;
		auto saved_user_id = s->value("user_id", user_id).toInt(&user_id_ok);
		user_id            = user_id_ok && Config::IsConfiguredUserIdValid(saved_user_id)
		                         ? saved_user_id
		                         : Config::DEFAULT_USER_ID;
		KYTY_CFG_GET(present_mode);
		gpu_index = s->value("gpu_index", -1).toInt();
		if (EnumToText(present_mode).isEmpty()) {
			present_mode = PresentMode::Mailbox;
		}
		KYTY_CFG_GET(fullscreen_enabled);
		KYTY_CFG_GET(readback_linear_images);
		vblank_frequency = s->value("vblank_frequency", vblank_frequency).toInt();
		console_language = s->value("console_language", console_language).toInt();
		if (console_language < 0 || console_language > MAX_CONSOLE_LANGUAGE) {
			console_language = DEFAULT_CONSOLE_LANGUAGE;
		}
		KYTY_CFG_GET(vulkan_validation_enabled);
		KYTY_CFG_GET(shader_validation_enabled);
		KYTY_CFG_GET(shader_optimization_type);
		KYTY_CFG_GET(shader_log_direction);
		KYTY_CFG_GET(shader_log_folder);
		KYTY_CFG_GET(command_buffer_dump_enabled);
		KYTY_CFG_GET(command_buffer_dump_folder);
		KYTY_CFG_GET(printf_direction);
		KYTY_CFG_GET(printf_output_file);
		KYTY_CFG_GET(profiler_enabled);
		KYTY_CFG_GET(renderdoc_enabled);
#if defined(_WIN32)
		red_zone_protection_enabled =
		    s->value("red_zone_protection_enabled", red_zone_protection_enabled).toBool();
#endif
		host_input_mapping = s->value("host_input_mapping", host_input_mapping).toStringList();
		elf                = s->value("elf", elf).toString();
		KYTY_CFG_GET_OR(pre_gen_enabled);
		KYTY_CFG_GET_OR(gpu_assisted_validation_enabled);
		KYTY_CFG_GET_OR(graphics_debug_dump_enabled);
		KYTY_CFG_GET_OR(spirv_debug_printf_enabled);
		KYTY_CFG_GET_OR(perf_stats_enabled);
		KYTY_CFG_GET_OR(gpu_labels_enabled);
		KYTY_CFG_GET_OR(sync_compute_enabled);
		KYTY_CFG_GET_OR(tessellation_enabled);
		KYTY_CFG_GET_OR(force_ui_mask_enabled);
		KYTY_CFG_GET_OR(pixel_quad_derivatives_enabled);
		KYTY_CFG_GET_OR(trace_dcc_enabled);
		KYTY_CFG_GET_OR(cmask_clear_disabled);
		KYTY_CFG_GET_OR(shader_loop_guard_hashes);
		KYTY_CFG_GET_OR(shader_loop_limit);
		KYTY_CFG_GET_OR(skip_cs_hashes);
		KYTY_CFG_GET_OR(skip_cs_addresses);
		KYTY_CFG_GET_OR(trace_nan_cs);
		KYTY_CFG_GET_OR(extra_environment);
	}

	// The emulator's KYTY_* variables for these settings, followed by extra_environment.
	[[nodiscard]] QList<EnvironmentVariable> EmulatorEnvironment() const {
		QList<EnvironmentVariable> variables;
		// The emulator turns these on only when the variable is exactly "1".
		const auto add_switch = [&variables](const char* name, bool enabled) {
			variables.push_back(
			    EnvironmentVariable {QString::fromLatin1(name), QStringLiteral("1"), enabled});
		};
		// These are on unless the variable is "0".
		const auto add_default_on = [&variables](const char* name, bool enabled) {
			variables.push_back(
			    EnvironmentVariable {QString::fromLatin1(name), QStringLiteral("0"), !enabled});
		};
		// An empty field leaves the variable unset, so the emulator uses its built-in default.
		const auto add_text = [&variables](const char* name, const QString& value) {
			const auto trimmed = value.trimmed();
			variables.push_back(
			    EnvironmentVariable {QString::fromLatin1(name), trimmed, !trimmed.isEmpty()});
		};

		add_switch("KYTY_PERF_STATS", perf_stats_enabled);
		add_switch("KYTY_GPU_LABELS", gpu_labels_enabled);
		add_switch("KYTY_SYNC_COMPUTE", sync_compute_enabled);
		add_switch("KYTY_ENABLE_TESSELLATION", tessellation_enabled);
		add_switch("KYTY_TRACE_DCC", trace_dcc_enabled);
		add_switch("KYTY_DISABLE_CMASK_CLEAR", cmask_clear_disabled);
		add_default_on("KYTY_FORCE_UI_MASK", force_ui_mask_enabled);
		add_default_on("KYTY_PIXEL_QUAD_DERIVATIVES", pixel_quad_derivatives_enabled);
		add_text("KYTY_SHADER_LOOP_GUARD_HASHES", shader_loop_guard_hashes);
		add_text("KYTY_SHADER_LOOP_LIMIT", shader_loop_limit);
		add_text("KYTY_SKIP_CS_HASH", skip_cs_hashes);
		add_text("KYTY_SKIP_CS_ADDRESS", skip_cs_addresses);
		add_text("KYTY_TRACE_NAN_CS", trace_nan_cs);
		static_cast<void>(ParseEnvironment(extra_environment, &variables, nullptr));
		return variables;
	}

	// Appends the NAME=VALUE entries of `text` (separated by ';' or new lines) to `out`. Returns
	// false if an entry is malformed; `invalid_entry` receives the first one, and valid entries are
	// still appended.
	static bool ParseEnvironment(const QString& text, QList<EnvironmentVariable>* out,
	                             QString* invalid_entry) {
		bool       valid   = true;
		const auto entries = QString(text)
		                         .replace(QLatin1Char('\n'), QLatin1Char(';'))
		                         .split(QLatin1Char(';'), Qt::SkipEmptyParts);
		for (const auto& entry: entries) {
			const auto trimmed = entry.trimmed();
			if (trimmed.isEmpty()) {
				continue;
			}
			const auto split = trimmed.indexOf(QLatin1Char('='));
			const auto name  = split > 0 ? trimmed.left(split).trimmed() : QString();
			if (!IsEnvironmentName(name)) {
				if (valid && invalid_entry != nullptr) {
					*invalid_entry = trimmed;
				}
				valid = false;
				continue;
			}
			if (out != nullptr) {
				out->push_back(EnvironmentVariable {name, trimmed.mid(split + 1).trimmed(), true});
			}
		}
		return valid;
	}

	// Letters, digits and '_', not starting with a digit. This also keeps names safe to write
	// unquoted into the Linux launch script.
	[[nodiscard]] static bool IsEnvironmentName(const QString& name) {
		if (name.isEmpty() || name.at(0).isDigit()) {
			return false;
		}
		for (const auto c: name) {
			if (c != QLatin1Char('_') && (c.unicode() >= 0x80 || !c.isLetterOrNumber())) {
				return false;
			}
		}
		return true;
	}
};

Q_DECLARE_METATYPE(Configuration*)

#endif /* LAUNCHER_INCLUDE_CONFIGURATION_H_ */
