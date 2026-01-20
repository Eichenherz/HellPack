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

#endif // !__HP_MATH_H__