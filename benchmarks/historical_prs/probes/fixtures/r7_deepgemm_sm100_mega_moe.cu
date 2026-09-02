#include <deep_gemm/impls/sm100_bf16_mega_moe.cuh>
#include <deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh>

using namespace deep_gemm;

static void instantiate_bf16() {
  auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_moe_impl<
      128,
      1024, 2048,
      8, 0,
      2,
      64, 128, 64,
      32,
      512,
      2,
      2048,
      128, 128, 128,
      120, 1,
      1.0e30f,
      false>);
  (void)ptr;
}

static void instantiate_fp8_fp4() {
  auto ptr = reinterpret_cast<void*>(&sm100_fp8_fp4_mega_moe_impl<
      128,
      1024, 2048,
      8, 0,
      2,
      64, 128, 128,
      32,
      128, 128,
      512,
      512,
      2,
      1024,
      128, 128, 128,
      120, 1,
      1.0e30f,
      false>);
  (void)ptr;
}

int main() {
  instantiate_bf16();
  instantiate_fp8_fp4();
  return 0;
}
