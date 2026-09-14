#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_

#include "common/abi.h"
#include "common/common.h"

#include <cstdint>

namespace Libs::Graphics {

class Presenter;

[[nodiscard]] Presenter& WindowInit(uint32_t width, uint32_t height);
void                     WindowRun();
void                     WindowShutdown();
[[nodiscard]] uint64_t   WindowGetPresentedFrameNum() noexcept;

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_WINDOW_H_ */
