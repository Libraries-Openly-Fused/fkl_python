// Minimal reproducer 1: ColorConversion<COLOR_BGR2GRAY>::build() does not exist
// because the alias resolves to FusedOperation_<...> (the raw Operations list),
// which only has UnaryType build() when ALL ops are Unary -- but the list
// contains the *Operation types*, not IOps, so ::build is missing entirely.
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/execution_model.h>
#include <fused_kernel/algorithms/algorithms.h>

using namespace fk;

int main() {
    // COLOR_RGB2GRAY works (plain RGB2Gray op):
    auto ok = ColorConversion<ColorConversionCodes::COLOR_RGB2GRAY, uchar3, uchar>::build();
    // COLOR_BGR2GRAY fails to compile:
    auto bad = ColorConversion<ColorConversionCodes::COLOR_BGR2GRAY, uchar3, uchar>::build();
    (void)ok; (void)bad;
    return 0;
}
