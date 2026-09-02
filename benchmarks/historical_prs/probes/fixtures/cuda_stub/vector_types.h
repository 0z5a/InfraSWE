#pragma once

#include <cstdint>

struct dim3 {
  unsigned int x, y, z;
  constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
      : x(x_), y(y_), z(z_) {}
};

#define INFRASWE_VECTOR1(name, type) struct name { type x; }
#define INFRASWE_VECTOR2(name, type) struct name { type x, y; }
#define INFRASWE_VECTOR3(name, type) struct name { type x, y, z; }
#define INFRASWE_VECTOR4(name, type) struct name { type x, y, z, w; }

INFRASWE_VECTOR1(char1, signed char);
INFRASWE_VECTOR2(char2, signed char);
INFRASWE_VECTOR3(char3, signed char);
INFRASWE_VECTOR4(char4, signed char);
INFRASWE_VECTOR1(uchar1, unsigned char);
INFRASWE_VECTOR2(uchar2, unsigned char);
INFRASWE_VECTOR3(uchar3, unsigned char);
INFRASWE_VECTOR4(uchar4, unsigned char);
INFRASWE_VECTOR1(short1, short);
INFRASWE_VECTOR2(short2, short);
INFRASWE_VECTOR3(short3, short);
INFRASWE_VECTOR4(short4, short);
INFRASWE_VECTOR1(ushort1, unsigned short);
INFRASWE_VECTOR2(ushort2, unsigned short);
INFRASWE_VECTOR3(ushort3, unsigned short);
INFRASWE_VECTOR4(ushort4, unsigned short);
INFRASWE_VECTOR1(int1, int);
INFRASWE_VECTOR2(int2, int);
INFRASWE_VECTOR3(int3, int);
INFRASWE_VECTOR4(int4, int);
INFRASWE_VECTOR1(uint1, unsigned int);
INFRASWE_VECTOR2(uint2, unsigned int);
INFRASWE_VECTOR3(uint3, unsigned int);
INFRASWE_VECTOR4(uint4, unsigned int);
INFRASWE_VECTOR1(long1, long);
INFRASWE_VECTOR2(long2, long);
INFRASWE_VECTOR3(long3, long);
INFRASWE_VECTOR4(long4, long);
INFRASWE_VECTOR1(ulong1, unsigned long);
INFRASWE_VECTOR2(ulong2, unsigned long);
INFRASWE_VECTOR3(ulong3, unsigned long);
INFRASWE_VECTOR4(ulong4, unsigned long);
INFRASWE_VECTOR1(longlong1, long long);
INFRASWE_VECTOR2(longlong2, long long);
INFRASWE_VECTOR3(longlong3, long long);
INFRASWE_VECTOR4(longlong4, long long);
INFRASWE_VECTOR1(ulonglong1, unsigned long long);
INFRASWE_VECTOR2(ulonglong2, unsigned long long);
INFRASWE_VECTOR3(ulonglong3, unsigned long long);
INFRASWE_VECTOR4(ulonglong4, unsigned long long);
INFRASWE_VECTOR1(float1, float);
INFRASWE_VECTOR2(float2, float);
INFRASWE_VECTOR3(float3, float);
INFRASWE_VECTOR4(float4, float);
INFRASWE_VECTOR1(double1, double);
INFRASWE_VECTOR2(double2, double);
INFRASWE_VECTOR3(double3, double);
INFRASWE_VECTOR4(double4, double);

#undef INFRASWE_VECTOR1
#undef INFRASWE_VECTOR2
#undef INFRASWE_VECTOR3
#undef INFRASWE_VECTOR4
