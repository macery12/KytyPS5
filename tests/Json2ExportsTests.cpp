#include "common/abi.h"
#include "loader/symbolDatabase.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace Libs::LibJson2 {
void InitNet_1_Json2(Loader::SymbolDatabase* symbols);
}

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "Json2ExportsTests: failed: %s\n", text);
		std::abort();
	}
}

const Loader::SymbolRecord* FindFunction(Loader::SymbolDatabase* symbols, const char* nid) {
	const auto* symbol = symbols->FindByNid(nid, Loader::SymbolType::Func);
	Check(symbol != nullptr, "Json2 export is registered");
	return symbol;
}

} // namespace

int main() {
	Loader::SymbolDatabase symbols;
	Libs::LibJson2::InitNet_1_Json2(&symbols);

	using ValueTypeCtor = void* (KYTY_SYSV_ABI*)(void*, uint32_t);
	using ValueClear = void (KYTY_SYSV_ABI*)(void*);
	using ValueGetType = uint32_t (KYTY_SYSV_ABI*)(const void*);
	using ValueSetNullAccessCallback = int32_t (KYTY_SYSV_ABI*)(void*, void*, void*);

	const auto value_type_ctor = reinterpret_cast<ValueTypeCtor>(
	    FindFunction(&symbols, "CbrT3dwDILo")->vaddr);
	const auto value_clear =
	    reinterpret_cast<ValueClear>(FindFunction(&symbols, "FIjXN2TkuTs")->vaddr);
	const auto value_get_type = reinterpret_cast<ValueGetType>(
	    FindFunction(&symbols, "SHtAad20YYM")->vaddr);
	const auto set_null_access_callback = reinterpret_cast<ValueSetNullAccessCallback>(
	    FindFunction(&symbols, "Xbl-LYVFNEE")->vaddr);

	alignas(8) std::array<std::byte, 32> value {};
	Check(value_type_ctor(value.data(), 5) == value.data(), "construct string value");
	Check(value_get_type(value.data()) == 5, "constructed value has string type");
	Check(set_null_access_callback(value.data(), nullptr, nullptr) == 0,
	      "null-access callback registration succeeds");
	value_clear(value.data());
	Check(value_get_type(value.data()) == 0, "clear resets value to null");

	return 0;
}
