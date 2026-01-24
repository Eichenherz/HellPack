#ifndef __HP_MATH_H__
#define __HP_MATH_H__

#ifdef __clang__
// NOTE: clang-cl on VS issue
#undef __clang__
#define _XM_NO_XMVECTOR_OVERLOADS_
#include <DirectXMath.h>
#define __clang__

#elif _MSC_VER >= 1916

#define _XM_NO_XMVECTOR_OVERLOADS_
#include <DirectXMath.h>

#endif

#include <cmath>

using float4x4 = DirectX::XMFLOAT4X4A;
using float3x3 = DirectX::XMFLOAT3X3;
using float4 = DirectX::XMFLOAT4;
using float3 = DirectX::XMFLOAT3;
using float2 = DirectX::XMFLOAT2;
using uint4 = DirectX::XMUINT4;
using uint3 = DirectX::XMUINT3;
using uint2 = DirectX::XMUINT2;
using int4 = DirectX::XMINT4;
using int3 = DirectX::XMINT3;
using int2 = DirectX::XMINT2;

struct u16x3
{
	u16 x, y, z;
};

#include <immintrin.h>

// NOTE: from https://stackoverflow.com/questions/17638487/minimum-of-4-sp-values-in-m128
inline __m128 _mm_hmin_ps( __m128 v )
{
	v = _mm_min_ps( v, _mm_shuffle_ps( v, v, _MM_SHUFFLE( 2, 1, 0, 3 ) ) );
	v = _mm_min_ps( v, _mm_shuffle_ps( v, v, _MM_SHUFFLE( 1, 0, 3, 2 ) ) );
	return v;
}

inline __m128 _mm_hmax_ps( __m128 v )
{
	v = _mm_max_ps( v, _mm_shuffle_ps( v, v, _MM_SHUFFLE( 2, 1, 0, 3 ) ) );
	v = _mm_max_ps( v, _mm_shuffle_ps( v, v, _MM_SHUFFLE( 1, 0, 3, 2 ) ) );
	return v;
}

inline float MinF32x8_SIMD( __m256 a, __m256 b )
{
	__m256 laneMin_f32x8 = _mm256_min_ps( a, b );
	__m128 lo = _mm256_castps256_ps128( laneMin_f32x8 );
	__m128 hi = _mm256_extractf128_ps( laneMin_f32x8, 1 );

	__m128 laneMin_f32x4 = _mm_min_ps( lo, hi );

	__m128 min_f32x4 = _mm_hmin_ps( laneMin_f32x4 );

	return _mm_cvtss_f32( min_f32x4 );
}

inline float MaxF32x8_SIMD( __m256 a, __m256 b )
{
	__m256 laneMax_f32x8 = _mm256_max_ps( a, b );
	__m128 lo = _mm256_castps256_ps128( laneMax_f32x8 );
	__m128 hi = _mm256_extractf128_ps( laneMax_f32x8, 1 );

	__m128 laneMax_f32x4 = _mm_max_ps( lo, hi );

	__m128 max_f32x4 = _mm_hmax_ps( laneMax_f32x4 );

	return _mm_cvtss_f32( max_f32x4 );
}

template<typename Vec>
struct aabb_t
{
	Vec min;
	Vec max;
};

inline aabb_t<float3> ComputeAabb( std::span<const float3> vertices )
{
	float3 min = vertices[ 0 ];
	float3 max = vertices[ 0 ];

	for( u64 vi = 1; vi < std::size( vertices ); ++vi )
	{
		min.x = std::min( min.x, vertices[ vi ].x );
		min.y = std::min( min.y, vertices[ vi ].y );
		min.z = std::min( min.z, vertices[ vi ].z );

		max.x = std::max( max.x, vertices[ vi ].x );
		max.y = std::max( max.y, vertices[ vi ].y );
		max.z = std::max( max.z, vertices[ vi ].z );
	}
	return { min, max };
}

inline aabb_t<float2> ComputeAabb( std::span<const float2> vertices )
{
	float2 min = vertices[ 0 ];
	float2 max = vertices[ 0 ];

	for( u64 vi = 1; vi < std::size( vertices ); ++vi )
	{
		min.x = std::min( min.x, vertices[ vi ].x );
		min.y = std::min( min.y, vertices[ vi ].y );

		max.x = std::max( max.x, vertices[ vi ].x );
		max.y = std::max( max.y, vertices[ vi ].y );
	}
	return { min, max };
}

inline aabb_t<float3> MergeAabbPair( const aabb_t<float3>& a, const aabb_t<float3>& b )
{
	return {
		.min = {
			std::min( a.min.x, b.min.x ),
			std::min( a.min.y, b.min.y ),
			std::min( a.min.z, b.min.z ),
		},
		.max = {
			std::max( a.max.x, b.max.x ),
			std::max( a.max.y, b.max.y ),
			std::max( a.max.z, b.max.z ),
		}
	};
}

inline aabb_t<float3> MergeAabbsMultiple( const std::ranges::forward_range auto& aabbs )
{
	aabb_t<float3> out = {
		.min = { +FLT_MAX, +FLT_MAX, +FLT_MAX },
		.max = { -FLT_MAX, -FLT_MAX, -FLT_MAX },
	};

	for( const auto& box : aabbs )
	{
		out = MergeAabbPair( out, { .min = box.min, .max = box.max } );
	}

	return out;
}

inline aabb_t<float3> MergeAabbs( const std::ranges::forward_range auto& aabbs )
{
	const u64 meshletCount = std::size( aabbs );

	HP_ASSERT( meshletCount );
	if( 1 == meshletCount )
	{
		return { .min = aabbs[ 0 ].min, .max = aabbs[ 0 ].max };
	}

	if( 2 == meshletCount )
	{
		return MergeAabbPair(
			{ .min = aabbs[ 0 ].min, .max = aabbs[ 0 ].max },
			{ .min = aabbs[ 1 ].min, .max = aabbs[ 1 ].max }
		);
	}

	return MergeAabbsMultiple( aabbs );
}

#endif // !__HP_MATH_H__