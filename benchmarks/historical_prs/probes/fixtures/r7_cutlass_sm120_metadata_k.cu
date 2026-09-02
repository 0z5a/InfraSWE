#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"

#ifndef PROBE_TILE_K
#define PROBE_TILE_K 256
#endif

using namespace cute;

using ElementA = cutlass::float_e2m1_t;
using ElementB = cutlass::float_e2m1_t;
using ElementPairA = cutlass::nv_float4_t<ElementA>;
using ElementPairB = cutlass::nv_float4_t<ElementB>;
using TileShape = Shape<_128, _128, Int<PROBE_TILE_K>>;
using ClusterShape = Shape<_1, _1, _1>;

static constexpr int AlignmentA = 16 * 8 * 2 / cutlass::sizeof_bits<ElementA>::value;
static constexpr int AlignmentB = 16 * 8 / cutlass::sizeof_bits<ElementB>::value;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120,
    cutlass::arch::OpClassBlockScaledSparseTensorOp,
    ElementPairA,
    cutlass::layout::RowMajor,
    AlignmentA,
    ElementPairB,
    cutlass::layout::ColumnMajor,
    AlignmentB,
    float,
    TileShape,
    ClusterShape,
    cutlass::gemm::collective::StageCountAuto,
    cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120>::CollectiveOp;

static_assert(sizeof(typename CollectiveMainloop::SharedStorage) > 0);

__global__ void force_instantiation() {}
