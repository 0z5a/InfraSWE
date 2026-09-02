#include <cute/numeric/arithmetic_tuple.hpp>

struct LeftValue {
  int value;
};

struct RightValue {
  int value;
};

constexpr bool operator==(LeftValue lhs, LeftValue rhs) {
  return lhs.value == rhs.value;
}

constexpr bool operator==(RightValue lhs, RightValue rhs) {
  return lhs.value == rhs.value;
}

using LeftBasis0 = cute::ScaledBasis<LeftValue, 0>;
using LeftBasis1 = cute::ScaledBasis<LeftValue, 1>;
using RightBasis1 = cute::ScaledBasis<RightValue, 1>;

constexpr LeftBasis0 left_one{LeftValue{1}};
constexpr LeftBasis0 left_one_again{LeftValue{1}};
constexpr LeftBasis0 left_two{LeftValue{2}};
constexpr LeftBasis1 other_basis_same_type{LeftValue{1}};
constexpr RightBasis1 other_basis_uncomparable_type{RightValue{1}};

static_assert(left_one == left_one_again, "equal basis and value must compare equal");
static_assert(!(left_one == left_two), "equal basis and unequal value must compare unequal");
static_assert(
    !(left_one == other_basis_same_type),
    "different bases with comparable values must compare unequal");
static_assert(
    !(left_one == other_basis_uncomparable_type),
    "different bases with uncomparable values must compile and compare unequal");

int main() {
  return (left_one == left_one_again) && !(left_one == other_basis_uncomparable_type)
      ? 0
      : 1;
}
