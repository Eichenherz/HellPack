#include <meshoptimizer.h>

#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

#include <span>
#include <ranges>
#include <type_traits>

#include <ankerl/unordered_dense.h>

#include <dds.h>

#include "core_types.h"
#include "hp_error.h"

#include "hell_pack.h"

// NOTE/TODO: float types are fwd def in mesh to be shared ! must use an internal folder or smth for math
#include "hp_math.h"
#include "hp_mesh.h"

#include "hp_material.h"
#include "hp_bvh_builder.h"
#include "hp_encoding.h"
#include "hp_bcn_compression.h"
#include "hp_serialization.h"
#include "mikkt_space.h"

#include "range_utils.h"

#include "gltf_loader.h"

#include "hp_types_internal.h"

raw_mesh ValidateAndNormalizeRawMesh( const raw_mesh& inRawMesh )
{
	HP_ASSERT( std::size( inRawMesh.indices ) != 0 );
	HP_ASSERT( ( std::size( inRawMesh.indices ) % 3 ) == 0 );
	auto it = std::ranges::max_element( inRawMesh.indices );
	HP_ASSERT( *it <= u16( -1 ) );
	HP_ASSERT( inRawMesh.materialIdx <= i32( u16( -1 ) ) );

	raw_mesh outRawMesh = {
		.name = std::move( inRawMesh.name ),
		.pos = std::move( inRawMesh.pos ),
		.normals = std::move( inRawMesh.normals ),
		.tans = std::move( inRawMesh.tans ),
		.uvs = std::move( inRawMesh.uvs ),
		.indices = std::move( inRawMesh.indices ),
		.materialIdx = inRawMesh.materialIdx
	};

	if( !std::size( outRawMesh.tans ) )
	{
		outRawMesh.tans = ComputeMikkTSpaceTangentsInplace( inRawMesh );
	}

	return outRawMesh;
}

struct meshlet_config
{
	float   coneWeight = 0.8f;
	u16		maxVertices = 64;
	u16		maxTriangles = 128;
};

struct rt_cluster_config
{
	float   fillWeight = 0.5f;
	u16		maxVertices = 64;
	u16		minTriangles = 16;
	u16		maxTriangles = 64;
};

template<typename T>
concept IsMeshletCfg = std::same_as<T, meshlet_config>;

template<typename T>
concept IsRtClusterCfg = std::same_as<T, rt_cluster_config>;

template<typename T>
concept IsClusterConfig = IsMeshletCfg<T> || IsRtClusterCfg<T>;

// TODO: no inplace remap !
void ReindexAndOptimizeMesh( raw_mesh& rawMesh )
{
	meshopt_Stream attrStreams[] = {
		{ .data = std::data( rawMesh.pos ), .size = std::size( rawMesh.pos ), .stride = sizeof( rawMesh.pos[ 0 ] ) },
		{ .data = std::data( rawMesh.normals ), .size = std::size( rawMesh.normals ), .stride = sizeof( rawMesh.normals[ 0 ] ) },
		{ .data = std::data( rawMesh.tans ), .size = std::size( rawMesh.tans ), .stride = sizeof( rawMesh.tans[ 0 ] ) },
		{ .data = std::data( rawMesh.uvs ), .size = std::size( rawMesh.uvs ), .stride = sizeof( rawMesh.uvs[ 0 ] ) },
	};
	std::vector<u32>& indices = rawMesh.indices;

	const u64 vtxCount = std::size( rawMesh.pos );
	const u64 idxCount = std::size( indices );

	std::vector<u32> remap( vtxCount );
	u64 newVtxCount = meshopt_generateVertexRemapMulti(
		std::data( remap ), std::data( indices ), idxCount, vtxCount, attrStreams, std::size( attrStreams ) );

	HP_ASSERT( newVtxCount <= vtxCount );
	if( newVtxCount != vtxCount )
	{
		meshopt_remapIndexBuffer( std::data( indices ), std::data( indices ), idxCount, std::data( remap ) );
		meshopt_remapVertexBuffer( std::data( rawMesh.pos ), std::data( rawMesh.pos ), vtxCount, 
								   sizeof( rawMesh.pos[ 0 ] ), std::data( remap ) );
		rawMesh.pos.resize( newVtxCount );
		meshopt_remapVertexBuffer( std::data( rawMesh.normals ), std::data( rawMesh.normals ), vtxCount, 
								   sizeof( rawMesh.normals[ 0 ] ), std::data( remap ) );
		rawMesh.normals.resize( newVtxCount );
		meshopt_remapVertexBuffer( std::data( rawMesh.tans ), std::data( rawMesh.tans ), vtxCount, 
								   sizeof( rawMesh.tans[ 0 ] ), std::data( remap ) );
		rawMesh.tans.resize( newVtxCount );
		meshopt_remapVertexBuffer( std::data( rawMesh.uvs ), std::data( rawMesh.uvs ), vtxCount, 
								   sizeof( rawMesh.uvs[ 0 ] ), std::data( remap ) );
		rawMesh.uvs.resize( newVtxCount );
	}

	meshopt_optimizeVertexCache( std::data( indices ), std::data( indices ), idxCount, newVtxCount );

	meshopt_optimizeVertexFetch( std::data( rawMesh.pos ), std::data( indices ), idxCount, std::data( rawMesh.pos ), 
								 newVtxCount, sizeof( rawMesh.pos[ 0 ] ) );
	meshopt_optimizeVertexFetch( std::data( rawMesh.normals ), std::data( indices ), idxCount, std::data( rawMesh.normals ), 
								 newVtxCount, sizeof( rawMesh.normals[ 0 ] ) );
	meshopt_optimizeVertexFetch( std::data( rawMesh.tans ), std::data( indices ), idxCount, std::data( rawMesh.tans ), 
								 newVtxCount, sizeof( rawMesh.tans[ 0 ] ) );
	meshopt_optimizeVertexFetch( std::data( rawMesh.uvs ), std::data( indices ), idxCount, std::data( rawMesh.uvs ), 
								 newVtxCount, sizeof( rawMesh.uvs[ 0 ] ) );
}

struct __meshopt_meshlets
{
	std::vector<meshopt_Meshlet> info;
	std::vector<u32> vertices;
	std::vector<u8> triangles;
};

template<IsClusterConfig Cfg>
__meshopt_meshlets MeshoptMakeClusters( 
	std::span<const float3> pos, 
	std::span<const u32> indices, 
	Cfg cfg 
) { 
	const u64 indexCount = std::size( indices );

	u64 triangleBound = cfg.maxTriangles;
	if constexpr( IsRtClusterCfg<Cfg> )
	{
		// NOTE( meshoptimizer ): use minTriangles to compute worst case bound
		triangleBound = cfg.minTriangles;
	}
	
	const u64 maxMeshletCount = meshopt_buildMeshletsBound( indexCount, cfg.maxVertices, triangleBound );
	std::vector<meshopt_Meshlet> meshlets( maxMeshletCount );
	std::vector<u32> mletVtx( indexCount );
	std::vector<u8> mletTris( indexCount );

	u64 meshletCount;
	if constexpr( IsMeshletCfg<Cfg> )
	{
		meshletCount = meshopt_buildMeshlets(
			&meshlets[ 0 ], &mletVtx[ 0 ], &mletTris[ 0 ], &indices[ 0 ], std::size( indices ),
			&pos[ 0 ].x, std::size( pos ), sizeof( pos[ 0 ] ), cfg.maxVertices, cfg.maxTriangles, cfg.coneWeight );
	}
	else if constexpr( IsRtClusterCfg<Cfg> )
	{
		meshletCount = meshopt_buildMeshletsSpatial(
			&meshlets[ 0 ], &mletVtx[ 0 ], &mletTris[ 0 ], &indices[ 0 ], std::size( indices ), &pos[ 0 ].x, 
			std::size( pos ), sizeof( pos[ 0 ] ), cfg.maxVertices, cfg.minTriangles, cfg.maxTriangles, cfg.fillWeight );
	}

	const meshopt_Meshlet& last = meshlets[ meshletCount - 1 ];

	meshlets.resize( meshletCount );
	mletVtx.resize( ( u64 ) last.vertex_offset + last.vertex_count );
	mletTris.resize( ( u64 ) last.triangle_offset + ( ( ( u64 ) last.triangle_count * 3 + 3 ) & ~3 ) );

	for( const meshopt_Meshlet& m : meshlets )
	{
		meshopt_optimizeMeshlet( &mletVtx[ m.vertex_offset ], &mletTris[ m.triangle_offset ],
								 m.triangle_count, m.vertex_count );
	}

	return { 
		.info = std::move( meshlets ), 
		.vertices = std::move( mletVtx ), 
		.triangles = std::move( mletTris ) 
	};
}

template<typename T> 
inline auto GetMeshletLocalAttrStream( 
	std::span<const T> meshAttrStream, 
	std::span<const u32> mletVtx, 
	u64 mletVtxOffset, 
	u64 mletVtxCount 
){
	std::vector<T> localStream( mletVtxCount );
	for( u64 vi = 0; vi < std::size( localStream ); ++vi )
	{
		localStream[ vi ] = meshAttrStream[ mletVtx[ vi + mletVtxOffset ] ];
	}

	return localStream;
}


std::vector<meshlet> PackMeshlets( const raw_mesh& rawMesh )
{
	__meshopt_meshlets meshlets = MeshoptMakeClusters( rawMesh.pos, rawMesh.indices, meshlet_config{} );

	std::span<const float3> pos = rawMesh.pos;
	std::span<const float3> norm = rawMesh.normals;
	std::span<const float4> tan = rawMesh.tans;
	std::span<const float2> uvs = rawMesh.uvs;

	std::vector<meshlet> packedMeshlets;
	packedMeshlets.reserve( std::size( meshlets.info ) );

	for( const meshopt_Meshlet& m : meshlets.info )
	{
		auto firstTriangleIt = std::cbegin( meshlets.triangles ) + m.triangle_offset;
		std::vector<u8> triangles = { firstTriangleIt, firstTriangleIt + m.triangle_count };

		std::vector<float3> mletPosStream = GetMeshletLocalAttrStream( pos, meshlets.vertices, m.vertex_offset, m.vertex_count );

		auto mletNormStream = GetMeshletLocalAttrStream( norm, meshlets.vertices, m.vertex_offset, m.vertex_count );
		auto mletTanStream = GetMeshletLocalAttrStream( tan, meshlets.vertices, m.vertex_offset, m.vertex_count );
		auto mletUvStream = GetMeshletLocalAttrStream( uvs, meshlets.vertices, m.vertex_offset, m.vertex_count );

		std::vector<packed_vtx> packedVtx( std::size( mletPosStream ) );
		for( u64 vi = 0; vi < m.vertex_count; ++vi )
		{
			float3 p = mletPosStream[ vi ];
			float3 n = mletNormStream[ vi ];
			float4 t = mletTanStream[ vi ];
			float2 uv = mletUvStream[ vi ];
			float2 octNormal = OctaNormalEncode( n );
			float tanAngle = EncodeTanToAngle( n, { t.x,t.y,t.z } );
			u8 tanSign = ( -1.0f == t.w ) ? 1 : 0;

			packedVtx[ vi ] = {
				.pos = p,
				.octNormal = octNormal, 
				.tanAngle = tanAngle, 
				.u = uv.x, .v = uv.y, 
				.tanSign = tanSign 
			};
		}

		const aabb_t<float3> aabb = ComputeAabb( mletPosStream );

		meshlet rtMeshlet = {
			.vertices = std::move( packedVtx ), 
			.triangles = std::move( triangles ),
			.aabbMin = aabb.min, 
			.aabbMax = aabb.max 
		};
		packedMeshlets.push_back( std::move( rtMeshlet ) );
	}

	return packedMeshlets;
}

std::vector<rt_cluster> PackClustersRaytracing( const raw_mesh& rawMesh )
{
	__meshopt_meshlets meshlets = MeshoptMakeClusters( rawMesh.pos, rawMesh.indices, rt_cluster_config{} );

	std::span<const float3> pos = rawMesh.pos;
	std::span<const float3> norm = rawMesh.normals;
	std::span<const float4> tan = rawMesh.tans;
	std::span<const float2> uvs = rawMesh.uvs;

	std::vector<rt_cluster> packedMeshlets;
	packedMeshlets.reserve( std::size( meshlets.info ) );

	for( const meshopt_Meshlet& m : meshlets.info )
	{
		auto firstTriangleIt = std::cbegin( meshlets.triangles ) + m.triangle_offset;
		std::vector<u8> triangles = { firstTriangleIt, firstTriangleIt + m.triangle_count };

		std::vector<float3> mletPosStream = GetMeshletLocalAttrStream( pos, meshlets.vertices, m.vertex_offset, m.vertex_count );

		const aabb_t<float3> aabb = ComputeAabb( mletPosStream );

		auto mletNormStream = GetMeshletLocalAttrStream( norm, meshlets.vertices, m.vertex_offset, m.vertex_count );
		auto mletTanStream = GetMeshletLocalAttrStream( tan, meshlets.vertices, m.vertex_offset, m.vertex_count );
		auto mletUvStream = GetMeshletLocalAttrStream( uvs, meshlets.vertices, m.vertex_offset, m.vertex_count );

		std::vector<vertex_attrs> packedAttrs( std::size( mletNormStream ) );
		for( u64 vi = 0; vi < m.vertex_count; ++vi )
		{
			float3 n = mletNormStream[ vi ];
			float4 t = mletTanStream[ vi ];
			float2 uv = mletUvStream[ vi ];
			float2 octNormal = OctaNormalEncode( n );
			float tanAngle = EncodeTanToAngle( n, { t.x,t.y,t.z } );
			u8 tanSign = ( -1.0f == t.w ) ? 1 : 0;

			packedAttrs[ vi ] = {
				.octNormal = octNormal, 
				.tanAngle = tanAngle, 
				.u = uv.x, .v = uv.y, 
				.tanSign = tanSign 
			};
		}

		rt_cluster rtMeshlet = {
			.positions = std::move( mletPosStream ), 
			.packedAttrs = std::move( packedAttrs ), 
			.triangles = std::move( triangles ),
			.aabbMin = aabb.min, 
			.aabbMax = aabb.max 
		};
		packedMeshlets.push_back( std::move( rtMeshlet ) );
	}

	return packedMeshlets;
}

std::vector<vertex_attrs> PackVertexAttributes( 
	std::span<const float3> normals, 
	std::span<const float4> tans, 
	std::span<const float2> uvs 
) {
	u64 vtxCount = std::size( normals );

	HP_ASSERT( ( vtxCount == std::size( tans ) ) && ( vtxCount == std::size( uvs ) ) );

	std::vector<vertex_attrs> packedAttrs;
	packedAttrs.resize( vtxCount );

	for( u64 vi = 0; vi < vtxCount; ++vi )
	{
		float3 n = normals[ vi ];
		float4 t = tans[ vi ];
		float2 uv = uvs[ vi ];
		float2 octNormal = OctaNormalEncode( n );
		float tanAngle = EncodeTanToAngle( n, { t.x,t.y,t.z } );
		u8 tanSign = ( -1.0f == t.w ) ? 1 : 0;

		packedAttrs[ vi ] = {
			.octNormal = octNormal, 
			.tanAngle = tanAngle, 
			.u = uv.x, .v = uv.y, 
			.tanSign = tanSign 
		};
	}

	return packedAttrs;
}

packed_mesh PackMesh( const raw_mesh& rawMesh, bvh_builder& bvhBuilder )
{
	std::vector<vertex_attrs> packedAttrs;
	
	if( !std::size( rawMesh.tans ) )
	{
		std::vector<float4> tans = ComputeMikkTSpaceTangentsInplace( rawMesh );
		packedAttrs = PackVertexAttributes( rawMesh.normals, tans, rawMesh.uvs );
	}
	else
	{
		packedAttrs = PackVertexAttributes( rawMesh.normals, rawMesh.tans, rawMesh.uvs );
	}

	const auto& positions = rawMesh.pos;
	const auto& indices = rawMesh.indices;

	std::vector<float3> meshPos( std::size( positions ) );
	std::vector<vertex_attrs> meshVtxAttr( std::size( packedAttrs ) );
	std::vector<u16> meshIdx( std::size( indices ) );
	
	std::vector<gpu_bvh2_node> blas;
	aabb_t<float3> aabb = {};

	const bool buildBvhThresholdReached = std::size( indices ) > 3 * MAX_LEAF_PRIM_COUNT;
	if( buildBvhThresholdReached )
	{
		auto triAabbView = TriangleAabbView( positions, indices );
		bvh_output bvh = bvhBuilder.BuildBvhOverPrimitives( triAabbView );

		blas = bvh.gpuNodes;
		aabb = bvh.topLevelAabb;

		std::vector<u32> permutedIndices = PermuteTrianglesByPrimitiveRemap( indices, bvh.primitiveIndices );
		std::vector<u32> remap = BuildVertexRemapFromPermutedIndices( permutedIndices, std::size( positions ) );

		meshopt_remapIndexBuffer( 
			std::data( permutedIndices ), std::data( permutedIndices ), std::size( permutedIndices ), std::data( remap ) );

		u64 vtxCount = std::size( meshPos );
		meshopt_remapVertexBuffer( 
			std::data( meshPos ), std::data( positions ), vtxCount, sizeof( positions[ 0 ] ), std::data( remap ) );
		meshopt_remapVertexBuffer( 
			std::data( meshVtxAttr ), std::data( packedAttrs ), vtxCount, sizeof( packedAttrs[ 0 ] ), std::data( remap ) );

		auto u16IndicesView = permutedIndices | std::views::transform( [] ( u32 idx ) { return ( u16 ) idx; } );
		std::ranges::copy( u16IndicesView, std::begin( meshIdx ) );
	}
	else
	{
		auto u16IndicesView = rawMesh.indices | std::views::transform( [] ( u32 idx ) { return ( u16 ) idx; } );
		
		std::ranges::copy( positions, std::begin( meshPos ) );
		std::ranges::copy( packedAttrs, std::begin( meshVtxAttr ) );
		std::ranges::copy( u16IndicesView, std::begin( meshIdx ) );

		aabb = ComputeAabb( positions );
	}
	
	return {
		.positions = std::move( meshPos ),
		.attrs = std::move( meshVtxAttr ),
		.blas = std::move( blas ),
		.aabb = aabb,
		.materialIdx = ( u16 ) rawMesh.materialIdx 
	};
}

using position_t = float3;

struct clustered_world_data
{
	std::vector<clustered_instance> instances;
	std::vector<meshlet_info> meshletInfoBuffer;
	std::vector<packed_vtx> meshletVertexBuffer;
	std::vector<index_t> meshletTriangleBuffer;
	std::vector<material_desc> materials;
	std::vector<u8> textureDdsBlob;
	std::vector<sampler_config> samplers;

	range64 AppendMeshlets( const std::ranges::forward_range auto& meshlets )
	{
		HP_ASSERT( std::size( meshletVertexBuffer ) < u32( -1 ) );
		HP_ASSERT( std::size( meshletTriangleBuffer ) < u32( -1 ) );

		u64 totalVertexCount = 0;
		u64 totalTrianlgeCount = 0;
		for( const meshlet& m : meshlets )
		{
			const u64 vertexCount = std::size( m.vertices );
			const u64 triangleCount = std::size( m.triangles );

			HP_ASSERT( ( vertexCount <= u8( -1 ) ) && ( triangleCount <= u8( -1 ) ) );

			totalVertexCount += vertexCount;
			totalTrianlgeCount += triangleCount;
		}
		meshletVertexBuffer.reserve( std::size( meshletVertexBuffer ) + totalVertexCount );
		meshletTriangleBuffer.reserve( std::size( meshletTriangleBuffer ) + totalTrianlgeCount );

		HP_ASSERT( std::size( meshletInfoBuffer ) <= u32( -1 ) );
		const u32 baseMeshletOffset = ( u32 ) std::size( meshletInfoBuffer );
		meshletInfoBuffer.reserve( baseMeshletOffset + std::size( meshlets ) );

		for( const meshlet& m : meshlets )
		{
			meshlet_info info = {
				.aabbMin = m.aabbMin,
				.aabbMax = m.aabbMax,
				.vertexOffset = ( u32 ) std::size( meshletVertexBuffer ),
				.triangleOffset = ( u32 ) std::size( meshletTriangleBuffer ),
				.vertexCount = ( u8 ) std::size( m.vertices ),
				.triangleCount = ( u8 ) std::size( m.triangles )
			};

			std::ranges::copy( m.vertices, std::back_inserter( meshletVertexBuffer ) );
			std::ranges::copy( m.triangles, std::back_inserter( meshletTriangleBuffer ) );
			meshletInfoBuffer.push_back( info );
		}

		HP_ASSERT( std::size( meshlets ) <= u32( -1 ) );
		return { .baseOffset = baseMeshletOffset, .count = ( u32 ) std::size( meshlets ) };
	}

	inline std::vector<u8> HellPackSerialize()
	{
		hellpack_serializble_buffer bufs[] = {
			{ instances, hellpack_entry_type::INST },
			{ meshletInfoBuffer, hellpack_entry_type::MLET },
			{ meshletVertexBuffer, hellpack_entry_type::VTX },
			{ meshletTriangleBuffer, hellpack_entry_type::TRI },
			{ materials, hellpack_entry_type::MTRL },
			{ textureDdsBlob, hellpack_entry_type::TEX },
			{ samplers, hellpack_entry_type::SAMP },
		};

		return MakeHellpackBlob( bufs );
	}
};

//struct rt_clustered_world_data
//{
//	std::vector<rt_clustered_instance> instances;
//	std::vector<gpu_bvh2_node> globalTlasBuffer;
//	std::vector<gpu_bvh2_node> globalClasBuffer;
//	std::vector<meshlet_info> meshletInfoBuffer;
//	std::vector<position_t> globalVertexPosBuffer;
//	std::vector<vertex_attrs> globalPackedVertexBuffer;
//	std::vector<u8> globalTriangleBuffer;
//
//	range64 AppendMeshlets( const std::ranges::forward_range auto& meshlets )
//	{
//		HP_ASSERT( std::size( globalVertexPosBuffer ) == std::size( globalPackedVertexBuffer ) );
//		HP_ASSERT( std::size( globalVertexPosBuffer ) < u32( -1 ) );
//		HP_ASSERT( std::size( globalTriangleBuffer ) < u32( -1 ) );
//		HP_ASSERT( std::size( globalClasBuffer ) < u32( -1 ) );
//
//		u64 totalVertexCount = 0;
//		u64 totalTrianlgeCount = 0;
//		for( const rt_cluster& m : meshlets )
//		{
//			const u64 vertexCount = std::size( m.positions );
//			const u64 triangleCount = std::size( m.triangles );
//
//			HP_ASSERT( vertexCount == std::size( m.packedAttrs ) );
//			HP_ASSERT( ( vertexCount <= u8( -1 ) ) && ( triangleCount <= u8( -1 ) ) );
//
//			totalVertexCount += vertexCount;
//			totalTrianlgeCount += triangleCount;
//		}
//		globalVertexPosBuffer.reserve( std::size( globalVertexPosBuffer ) + totalVertexCount );
//		globalPackedVertexBuffer.reserve( std::size( globalPackedVertexBuffer ) + totalVertexCount );
//		globalTriangleBuffer.reserve( std::size( globalTriangleBuffer ) + totalTrianlgeCount );
//
//		HP_ASSERT( std::size( meshletInfoBuffer ) <= u32( -1 ) );
//		const u32 baseMeshletOffset = ( u32 ) std::size( meshletInfoBuffer );
//		meshletInfoBuffer.reserve( baseMeshletOffset + std::size( meshlets ) );
//
//		for( const rt_cluster& m : meshlets )
//		{
//			meshlet_info info = {
//				.aabbMin = m.aabbMin,
//				.aabbMax = m.aabbMax,
//				.vertexOffset = ( u32 ) std::size( globalVertexPosBuffer ),
//				.triangleOffset = ( u32 ) std::size( globalTriangleBuffer ),
//				.vertexCount = ( u8 ) std::size( m.positions ),
//				.triangleCount = ( u8 ) std::size( m.triangles )
//			};
//
//			std::ranges::copy( m.positions, std::back_inserter( globalVertexPosBuffer ) );
//			std::ranges::copy( m.packedAttrs, std::back_inserter( globalPackedVertexBuffer ) );
//			std::ranges::copy( m.triangles, std::back_inserter( globalTriangleBuffer ) );
//			meshletInfoBuffer.push_back( info );
//		}
//
//		HP_ASSERT( std::size( meshlets ) <= u32( -1 ) );
//		return { .baseOffset = baseMeshletOffset, .count = ( u32 ) std::size( meshlets ) };
//	}
//
//	inline std::vector<u8> HellPackSerialize()
//	{
//		hellpack_serializble_buffer bufs[] = {
//			 instances,
//			 globalTlasBuffer,
//			 globalClasBuffer,
//			 meshletInfoBuffer,
//			 globalVertexPosBuffer,
//			 globalPackedVertexBuffer,
//			 globalTriangleBuffer,
//		};
//
//		return MakeHellpackBlob( bufs );
//	}
//};
//
//struct rt_world_data
//{
//	std::vector<mesh_instance> instances;
//	std::vector<gpu_bvh2_node> globalTlasBuffer;
//	std::vector<gpu_bvh2_node> globalBlasBuffer;
//	std::vector<position_t> globalVertexPosBuffer;
//	std::vector<vertex_attrs> globalPackedVertexBuffer;
//	std::vector<u16> globalTriangleBuffer;
//
//	rt_mesh_desc AppendMesh( const packed_mesh& mesh )
//	{
//		HP_ASSERT( std::size( globalVertexPosBuffer ) == std::size( globalPackedVertexBuffer ) );
//		HP_ASSERT( std::size( globalVertexPosBuffer ) < u32( -1 ) );
//		HP_ASSERT( std::size( globalTriangleBuffer ) < u32( -1 ) );
//		HP_ASSERT( std::size( globalBlasBuffer ) < u32( -1 ) );
//
//		HP_ASSERT( std::size( mesh.blas ) < u16( -1 ) );
//		HP_ASSERT( std::size( mesh.positions ) < u16( -1 ) );
//		HP_ASSERT( std::size( mesh.indices ) < u16( -1 ) );
//
//		u32 baseBvhNodeOffset = std::size( globalBlasBuffer );
//		u32 baseVertexOffset = std::size( globalVertexPosBuffer );
//		u32 baseIndexOffset = std::size( globalTriangleBuffer );
//
//		u16 blasNodeCount = std::size( mesh.blas );
//		u16 vertexCount = std::size( mesh.positions );
//		u16 indexCount = std::size( mesh.indices );
//
//		globalBlasBuffer.reserve( baseBvhNodeOffset + blasNodeCount );
//		globalVertexPosBuffer.reserve( baseVertexOffset + vertexCount );
//		globalPackedVertexBuffer.reserve( baseVertexOffset + vertexCount );
//		globalTriangleBuffer.reserve( baseIndexOffset + indexCount );
//
//		std::ranges::copy( mesh.blas, std::back_inserter( globalBlasBuffer ) );
//		std::ranges::copy( mesh.positions, std::back_inserter( globalVertexPosBuffer ) );
//		std::ranges::copy( mesh.attrs, std::back_inserter( globalPackedVertexBuffer ) );
//		std::ranges::copy( mesh.indices, std::back_inserter( globalTriangleBuffer ) );
//
//		return { 
//			.bvhRoot = baseBvhNodeOffset,
//			.baseVertexOffset = baseVertexOffset,
//			.baseIndexOffset = baseIndexOffset,
//			.bvhNodeCount = blasNodeCount,
//			.vertexCount = vertexCount,
//			.indexCount = indexCount
//		};
//	}
//
//	inline std::vector<u8> HellPackSerialize()
//	{
//		hellpack_serializble_buffer bufs[] = {
//			instances,
//			globalTlasBuffer,
//			globalBlasBuffer,
//			globalVertexPosBuffer,
//			globalPackedVertexBuffer,
//			globalTriangleBuffer,
//		};
//
//		return MakeHellpackBlob( bufs );
//	}
//};

inline auto InstanceTlasAabbView( const std::ranges::forward_range auto& instances )
{
	return instances | std::views::transform( 
		[] ( const auto& i ) 
		{ 
			return TransformAABB( i.aabbMin, i.aabbMax, i.toWorld.t, i.toWorld.r, i.toWorld.s );
		} );
};

using dds_texture = std::vector<u8>;

constexpr bc_format_t DxgiToBcFormat( dds::DXGI_FORMAT dxgiFmt )
{
	using namespace dds;
	switch( dxgiFmt )
	{
	case DXGI_FORMAT_BC5_TYPELESS:
	case DXGI_FORMAT_BC5_UNORM:
	case DXGI_FORMAT_BC5_SNORM:
		return bc_format_t::BC5_RG;

	case DXGI_FORMAT_BC7_TYPELESS:
	case DXGI_FORMAT_BC7_UNORM:
	case DXGI_FORMAT_BC7_UNORM_SRGB:
		return bc_format_t::BC7_RGBA;

	default:
		HP_ASSERT( 0 && "Unimplement fmt" );
		return ( bc_format_t ) 0xFF;
	}
}

struct compression_job
{
	dds_texture tex;
	std::span<const u8> src;
	dds::DXGI_FORMAT fmt;
	u16 width;
	u16 height;

	void Execute()
	{
		bc_format_t bcnFmt = DxgiToBcFormat( fmt );
		// NOTE: these allocate memory !
		bcn_compression_result bcn = CompressRGBA8ToBCn( src, width, height, bcnFmt );

		tex.resize( sizeof( dds::Header ) + std::size( bcn.data ) );
		dds::write_header( &tex[ 0 ], fmt, width, height );
		std::memcpy( &tex[ 0 ] + sizeof( dds::Header ), &bcn.data[ 0 ], std::size( bcn.data ) );
	}
};

using compression_job_map_t = ankerl::unordered_dense::map<u16, compression_job>;

inline u64 GetTotalTexutreDataSizeInBytes( const compression_job_map_t& jobMap )
{
	u64 byteCount = 0;
	for( auto& [_, job] : jobMap )
	{
		byteCount += std::size( job.tex );
	}

	return byteCount;
}

compression_job_map_t
PrepareBcnCompressionBatch( std::span<const raw_material_info> materials, std::span<const raw_image_view> imageViews )
{
	HP_ASSERT( std::size( imageViews ) < u16( INVALID_IDX ) );

	compression_job_map_t jobsMap;

	auto ProcessImageView = [&] ( u16 idx, dds::DXGI_FORMAT fmt )
	{
		if( !IsIndexValid( idx ) ) return;
		if( jobsMap.find( idx ) != std::cend( jobsMap ) ) return;

		const raw_image_view& imgView = imageViews[ idx ];
		jobsMap.emplace( idx, compression_job{
			.src = imgView.data, .fmt = fmt, . width = imgView.metadata.width, .height = imgView.metadata.height } );
	};

	// NOTE: GLTF conventions
	for( const raw_material_info& material : materials )
	{
		ProcessImageView( material.baseColorIdx, dds::DXGI_FORMAT_BC7_UNORM_SRGB );
		ProcessImageView( material.normalIdx, dds::DXGI_FORMAT_BC5_UNORM );
		ProcessImageView( material.metallicRoughnessIdx, dds::DXGI_FORMAT_BC7_UNORM );
		//ProcessImageView( material.occlusionIdx, bc_format_t::BC7_RGBA );
		// NOTE: currently not suporting ambient occlusion which must be packed into MR
		HP_ASSERT( !IsIndexValid( material.occlusionIdx ) );
		ProcessImageView( material.emissiveIdx, dds::DXGI_FORMAT_BC7_UNORM_SRGB );
	}

	return jobsMap;
}

inline void ExecuteJobsParallel( std::vector<std::thread>& threadPool, std::function<void()> func, u64 threadCount )
{
	HP_ASSERT( std::size( threadPool ) == 0 );

	for( u64 ti = 0; ti < threadCount; ++ti )
	{
		threadPool.emplace_back( func );
	}
}

inline void WaitThreadPoolDone( std::vector<std::thread>& threadPool )
{
	for( auto& t : threadPool ) t.join();
}

template<typename K, typename V>
inline V MapGetIfExistsElseDefault( const ankerl::unordered_dense::map<K, V>& map, K idx )
{
	auto it = map.find( idx );
	if( it != std::cend( map ) ) return it->second;
	return {};
}

constexpr bool CHECK_SERIALIZATION_RESULT = true;

struct raw_mesh_desc
{
	float3	aabbMin;
	float3	aabbMax;
	u32		baseMeshletOffset;
	u16		meshletCount;
	u16		materialIdx;
};

int main()
{
	const std::string gltfFilePath = "D:/3d models/nightclub_futuristic_pub_ambience_asset.glb";
	HP_ASSERT( fs::exists( gltfFilePath ) );

	gltf_loader gltf = { gltfFilePath };

	// TODO: ensure we keep the same indexing as tinygltf provides !!!!
	std::vector<raw_node> rawNodes = gltf.ProcessNodes();
	std::vector<raw_mesh> rawMeshes = gltf.ProcessMeshes();
	std::vector<sampler_config> samplers = gltf.ProcessSamplers();
	std::vector<raw_material_info> rawMaterials = gltf.ProcessMaterials();
	std::vector<raw_image_view> imageViews = gltf.ProcessImages();

	compression_job_map_t texBatch = PrepareBcnCompressionBatch( rawMaterials, imageViews );

	std::vector<std::thread> tasks;
	std::atomic<u32> taskCounter = { 0 };

	auto jobs = texBatch | std::views::values;
	auto WorkerLoop = [&] ()
	{
		u32 currentJobIdx = taskCounter.fetch_add( 1 );
		if( std::size( texBatch ) <= currentJobIdx ) return;

		jobs[ currentJobIdx ].Execute();
	};

	ExecuteJobsParallel( tasks, WorkerLoop, std::thread::hardware_concurrency() );

	clustered_world_data worldData = {};

	//bvh_builder bvhBuilder = {};

	std::vector<raw_mesh_desc> meshDesc;
	meshDesc.reserve( std::size( rawMeshes ) );
	for( const raw_mesh& mesh : rawMeshes )
	{
		// NOTE: it moves stuff
		raw_mesh validatedRawMesh = ValidateAndNormalizeRawMesh( mesh );

		std::vector<meshlet> meshlets = PackMeshlets( validatedRawMesh );
		auto aabbView = meshlets | std::views::transform( 
			[] ( const meshlet& m ) { return aabb_t<float3>{ .min = m.aabbMin, .max = m.aabbMax }; } );

		aabb_t<float3> aabb = MergeAabbs( aabbView );

		range64 meshletsRange = worldData.AppendMeshlets( meshlets );

		meshDesc.push_back( {
			.aabbMin = aabb.min,
			.aabbMax = aabb.max,
			.baseMeshletOffset = ( u32 ) meshletsRange.baseOffset,
			.meshletCount = ( u16 ) meshletsRange.count,
			.materialIdx = ( u16 ) validatedRawMesh.materialIdx
		} );
	}

	worldData.instances.reserve( std::size( rawNodes ) );
	for( const raw_node& n : rawNodes )
	{
		if( INVALID_IDX == n.meshIdx ) continue;

		const raw_mesh_desc& md = meshDesc[ n.meshIdx ];

		worldData.instances.push_back( {
			.toWorld = n.toWorld,
			.aabbMin = md.aabbMin,
			.aabbMax = md.aabbMax,
			.baseMeshletOffset = md.baseMeshletOffset,
			.meshletCount = md.meshletCount,
			.materialIdx = md.meshletCount
		} );
	}

	WaitThreadPoolDone( tasks );

	ankerl::unordered_dense::map<u16, range64> texDataRemap;
	worldData.textureDdsBlob.reserve( GetTotalTexutreDataSizeInBytes( texBatch ) );
	for( auto&[ idx, job ] : texBatch )
	{
		range64 thisRange = { .baseOffset = std::size( worldData.textureDdsBlob ), .count = std::size( job.tex ) };
		texDataRemap.emplace( idx, thisRange );
		std::ranges::copy( job.tex, std::back_inserter( worldData.textureDdsBlob ) );
	}

	worldData.materials.reserve( std::size( rawMaterials ) );
	for( const raw_material_info& material : rawMaterials )
	{
		worldData.materials.push_back( {
			.baseColor = MapGetIfExistsElseDefault( texDataRemap, material.baseColorIdx ),
			.metallicRoughness = MapGetIfExistsElseDefault( texDataRemap, material.metallicRoughnessIdx ),
			.normal = MapGetIfExistsElseDefault( texDataRemap, material.normalIdx ),
			.emissive = MapGetIfExistsElseDefault( texDataRemap, material.emissiveIdx ),
			.baseColFactor = material.baseColFactor,
			.metallicFactor = material.metallicFactor,
			.roughnessFactor = material.metallicFactor,
			.alphaCutoff = material.alphaCutoff,
			.emissiveFactor = material.emissiveFactor,
			.samplerIdx = material.samplerIdx,
			.alphaMode = material.alphaMode
		} );
	}

	worldData.samplers = std::move( samplers );

	std::vector<u8> blob = worldData.HellPackSerialize();
	WriteFileBinary( "D:/3d models/nightclub_futuristic_pub_ambience_asset.hllp", blob );

	if constexpr( CHECK_SERIALIZATION_RESULT )
	{
		auto blob = ReadFileBinary( "D:/3d models/nightclub_futuristic_pub_ambience_asset.hllp" );

		hellpack_view hellpackView = { blob };

		hellpack_serializble_buffer bufs[] = {
			{ worldData.instances, hellpack_entry_type::INST },
			{ worldData.meshletInfoBuffer, hellpack_entry_type::MLET },
			{ worldData.meshletVertexBuffer, hellpack_entry_type::VTX },
			{ worldData.meshletTriangleBuffer, hellpack_entry_type::TRI },
			{ worldData.materials, hellpack_entry_type::MTRL },
			{ worldData.textureDdsBlob, hellpack_entry_type::TEX },
			{ worldData.samplers, hellpack_entry_type::SAMP },
		};

		for( u64 bi = 0; bi < std::size( bufs ); ++bi )
		{
			const auto& buf = bufs[ bi ];
			byte_view bv = hellpackView.Bytes( buf.typeOf );
			HP_ASSERT( ByteEqual( bv, buf.data ) );
		}
	}

	return 0;
}

