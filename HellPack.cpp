#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#include <tinygltf/tiny_gltf.h>
#include <meshoptimizer.h>

#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

#include <span>
#include <ranges>

#include "core_types.h"
#include "hp_error.h"
#include "hp_math.h"
#include "hp_types.h"
#include "hp_bvh.h"
#include "hp_encoding.h"

#include "mikkt_space.h"

inline packed_trs XM_CALLCONV XMComposePackedTRS( packed_trs a, packed_trs b )
{
	using namespace DirectX;

	XMVECTOR aT = XMLoadFloat3( &a.t );
	XMVECTOR aR = XMLoadFloat4( &a.r );
	XMVECTOR aS = XMLoadFloat3( &a.s );

	XMVECTOR bT = XMLoadFloat3( &b.t );
	XMVECTOR bR = XMLoadFloat4( &b.r );
	XMVECTOR bS = XMLoadFloat3( &b.s );

	float3 outT;
	XMStoreFloat3( &outT, XMVectorAdd( aT, bT ) );
	float4 outR;
	XMStoreFloat4( &outR, XMQuaternionMultiply( aR, bR ) ); // World = Parent * Local
	float3 outS;
	XMStoreFloat3( &outS, XMVectorMultiply( aS, bS ) );

	return { .t = outT, .r = outR, .s = outS };
}

struct gltf_raw_attr_stream
{
	const u8* data;
	u64 elemCount;
	u64 componentCount;
	u64 componentByteSize;
	u64 strideBytes; 

	constexpr u64 size() const noexcept
	{
		return elemCount;
	}
};

template<typename T>
struct gltf_typed_attr_stream : gltf_raw_attr_stream
{
	struct iterator
	{
		using iterator_category = std::input_iterator_tag;
		using value_type = T;
		using difference_type = std::ptrdiff_t;
		using pointer = void;
		using reference = T;

		const u8* ptr;
		u64 strideBytes;

		T operator*() const
		{
			T out;
			std::memcpy( &out, ptr, sizeof( T ) );
			return out;
		}

		iterator& operator++()
		{
			ptr += strideBytes;
			return *this;
		}

		iterator operator++( int )
		{
			iterator tmp = *this;
			++( *this );
			return tmp;
		}

		bool operator==( const iterator& rhs ) const { return ptr == rhs.ptr; }
		bool operator!=( const iterator& rhs ) const { return ptr != rhs.ptr; }
	};

	iterator begin() const
	{
		if( 0 == elemCount ) return { nullptr, strideBytes };

		HP_ASSERT( sizeof( T ) == ( componentCount * componentByteSize ) );
		HP_ASSERT( strideBytes >= sizeof( T ) );
		return { data, strideBytes };
	}

	iterator end() const
	{
		if( 0 == elemCount ) return { nullptr, strideBytes };

		HP_ASSERT( sizeof( T ) == ( componentCount * componentByteSize ) );
		HP_ASSERT( strideBytes >= sizeof( T ) );
		return { data + elemCount * strideBytes, strideBytes };
	}
};

template<typename T>
concept TinyGltfTextureInfoConcept = requires( T a ) {
	{ a.index } -> std::convertible_to<i32>;
	{ a.texCoord } -> std::convertible_to<i32>;
};

enum class gltf_img_components : u8
{
	UNKNOWN = 0,
	R = 1,
	RG = 2,
	RGB = 3,
	RGBA = 4
};

enum class gltf_img_bit_depth : u8
{
	UNKNOWN = 0,
	B8  = 8,
	B16 = 16,
	B32 = 32
};

enum class gltf_img_pixel_type : u8
{
	UNKNOWN = 0,
	UBYTE,
	USHORT,
	FLOAT32
};

struct gltf_image_metadata
{
	i32 width;
	i32 height;
	gltf_img_components component;
	gltf_img_bit_depth  bits;
	gltf_img_pixel_type pixelType;
};

struct gltf_texture
{
	i32 imageIdx = -1;
	i32 samplerIdx = -1;
};

struct gltf_image
{
	std::span<const u8> data;
	gltf_image_metadata metadata;
};

enum alpha_mode : u8
{
	ALPHA_MODE_OPAQUE,
	ALPHA_MODE_MASK,
	ALPHA_MODE_BLEND,
};

enum sampler_filter_mode_flags : u8
{
	FILTER_NEAREST = 1 << 0,
	FILTER_LINEAR = 1 << 1,
	FILTER_NEAREST_MIPMAP_NEAREST = 1 << 2,
	FILTER_LINEAR_MIPMAP_NEAREST = 1 << 3,
	FILTER_NEAREST_MIPMAP_LINEAR = 1 << 4,
	FILTER_LINEAR_MIPMAP_LINEAR = 1 << 5,
};

enum sampler_wrap_mode_flags : u8
{
	WRAP_CLAMP_TO_EDGE        = 1 << 0,
	WRAP_MIRRORED_REPEAT      = 1 << 1,
	WRAP_REPEAT               = 1 << 2,
};

struct sampler_config
{
	u32 filterModeS : 8;
	u32 filterModeT : 8;
	u32 wrapModeS : 8;
	u32 wrapModeT : 8;

	inline bool operator==( const sampler_config& rhs ) const
	{
		return filterModeS == rhs.filterModeS
			&& filterModeT == rhs.filterModeT
			&& wrapModeS == rhs.wrapModeS
			&& wrapModeT == rhs.wrapModeT;
	}
	inline bool operator!=( const sampler_config& rhs ) const
	{
		return !( *this == rhs );
	}
};

constexpr sampler_config DEFAULT_SAMPLER = {
	.filterModeS = sampler_filter_mode_flags::FILTER_LINEAR,
	.filterModeT = sampler_filter_mode_flags::FILTER_LINEAR,
	.wrapModeS = sampler_wrap_mode_flags::WRAP_REPEAT,
	.wrapModeT = sampler_wrap_mode_flags::WRAP_REPEAT
};

struct gltf_processor
{
	tinygltf::Model model;

	gltf_processor( std::string_view filePath )
	{
		std::string err, warn;
		if( tinygltf::TinyGLTF loader; !loader.LoadASCIIFromFile(
			&model, &err, &warn, std::string{ filePath }, tinygltf::SectionCheck::REQUIRE_ALL ) )
		{
			std::cout<< std::format("TinyGLTF LoadASCIIFromFile error: {}\nwarn: {}\n", err, warn );
			if( !loader.LoadBinaryFromFile(
				&model, &err, &warn, std::string{ filePath }, tinygltf::SectionCheck::REQUIRE_ALL ) )
			{
				std::cout<< std::format("TinyGLTF LoadBinaryFromFile error: {}\n warn: {}\n", err, warn );
				abort();
			}
		}

		HP_ASSERT( 1 == std::size( model.scenes ) );
		std::cout << "Successfully loaded the file.\n";
	}

	std::vector<raw_node> ProcessNodes() const
	{
		const std::vector<tinygltf::Node>& nodes = model.nodes;

		struct queued_node
		{
			packed_trs parentTRS;
			u32 nodeIdx;
		};
		std::vector<queued_node> nodeQueue;
		// NOTE: we need this bc our tree is flattened
		std::vector<bool> visited( std::size( nodes ), false );

		std::vector<raw_node> flatNodes;
		flatNodes.reserve( std::size( nodes ) );

		for( u32 nodeIdx = 0; nodeIdx < std::size( nodes ); ++nodeIdx )
		{
			if( visited[ nodeIdx ] ) continue;

			const tinygltf::Node& n = nodes[ nodeIdx ];

			packed_trs pkTrs = GetTrsFromNode( n );

			nodeQueue.push_back( { pkTrs, nodeIdx } );
			while( std::size( nodeQueue ) > 0 )
			{
				queued_node curr = nodeQueue.back();
				nodeQueue.pop_back();

				const tinygltf::Node& currentNode = nodes[ curr.nodeIdx ];
				packed_trs currentTrs = GetTrsFromNode( currentNode );

				packed_trs parentTrs = XMComposePackedTRS( curr.parentTRS, currentTrs );
				flatNodes.push_back( { parentTrs, currentNode.mesh } );
				visited[ curr.nodeIdx ] = true;

				for( u32 childNodeIdx : currentNode.children )
				{
					nodeQueue.push_back( { parentTrs, childNodeIdx } );
				}
			}
		}

		return flatNodes;
	}

	std::vector<raw_mesh> ProcessMeshes() const
	{
		u64 meshPrimitiveCount = 0;
		for( const tinygltf::Mesh& m : model.meshes )
		{
			meshPrimitiveCount += std::size( m.primitives );
		}

		std::vector<raw_mesh> meshesOut;
		meshesOut.reserve( meshPrimitiveCount );
		for( const tinygltf::Mesh& m : model.meshes )
		{
			for( const tinygltf::Primitive& primMesh : m.primitives )
			{
				// NOTE: we impose only indexed meshes
				HP_ASSERT( -1 != primMesh.indices );
				std::vector<u32> normalizedIndexBuffer = GetIndexBufferFromStream( model.accessors[ primMesh.indices ] );
				
				// NOTE: gltf guarentees that all present attr streams have the same element count 
				// NOTE: gltf mandates this stream be present
				const gltf_typed_attr_stream<float3> posStream = {
					GetRawAttributeStream( *GetAccessorByName( "POSITION", primMesh ) ) };
				
				raw_mesh mesh = {
					.name = m.name.c_str(),
					.pos = { std::cbegin( posStream ), std::cend( posStream ) },
					.indices = std::move( normalizedIndexBuffer ),
					.materialIdx = primMesh.material
				};

				if( const tinygltf::Accessor* pAccessor = GetAccessorByName( "NORMAL", primMesh ); pAccessor )
				{
					const gltf_typed_attr_stream<float3> normStream = { GetRawAttributeStream( *pAccessor ) };
					mesh.normals = { std::cbegin( normStream ), std::cend( normStream ) };
				}
				if( const tinygltf::Accessor* pAccessor = GetAccessorByName( "TANGENT", primMesh ); pAccessor )
				{
					const gltf_typed_attr_stream<float4> tanStream = { GetRawAttributeStream( *pAccessor ) };
					mesh.tans = { std::cbegin( tanStream ), std::cend( tanStream ) };
				}
				if( const tinygltf::Accessor* pAccessor = GetAccessorByName( "TEXCOORD_0", primMesh ); pAccessor )
				{
					const gltf_typed_attr_stream<float2> uvStream = { GetRawAttributeStream( *pAccessor ) };
					mesh.uvs = { std::cbegin( uvStream ), std::cend( uvStream ) };
				}
				meshesOut.emplace_back( mesh );
			}
		}

		return meshesOut;
	}

	//std::vector<sampler_config> ProcessSamplers() const
	//{
	//	std::vector<sampler_config> samplersOut;
	//	samplersOut.reserve( std::size( model.samplers ) );
	//	for( const tinygltf::Sampler& sampler : model.samplers )
	//	{
	//		sampler_config samplerConfig = {
	//			.filterModeS = GltfFilterToFlags( sampler.minFilter ),
	//			.filterModeT = GltfFilterToFlags( sampler.magFilter ),
	//			.wrapModeS = GltfWrapToFlags( sampler.wrapS ),
	//			.wrapModeT = GltfWrapToFlags( sampler.wrapT )
	//		};
	//		samplersOut.push_back( samplerConfig );
	//	}
	//	if( std::size( model.samplers ) == 0 )
	//	{
	//		samplersOut.push_back( DEFAULT_SAMPLER );
	//	}
	//
	//	return samplersOut;
	//}

	//std::vector<material_info> ProcessMaterials() const
	//{
	//	std::vector<material_info> materialsOut;
	//	materialsOut.reserve( std::size( model.materials ) );
	//	for( const tinygltf::Material& material : model.materials )
	//	{
	//		const tinygltf::PbrMetallicRoughness& pbrInfo = material.pbrMetallicRoughness;
	//
	//		material_info metadata = {
	//			.baseColFactor = {
	//				( float ) pbrInfo.baseColorFactor[ 0 ],
	//				( float ) pbrInfo.baseColorFactor[ 1 ],
	//				( float ) pbrInfo.baseColorFactor[ 2 ],
	//				( float ) pbrInfo.baseColorFactor[ 3 ]
	//		},
	//			.metallicFactor = ( float ) pbrInfo.metallicFactor,
	//			.roughnessFactor = ( float ) pbrInfo.roughnessFactor,
	//			.alphaCutoff = ( float ) material.alphaCutoff,
	//			.emissiveFactor = {
	//				( float ) material.emissiveFactor[ 0 ],
	//				( float ) material.emissiveFactor[ 1 ],
	//				( float ) material.emissiveFactor[ 2 ]
	//		},
	//			.alphaMode = GltfAlphaModeToEnum( material.alphaMode )
	//		};
	//
	//		ankerl::unordered_dense::set<i32> samplers;
	//		const gltf_texture normalTex = ProcessTexture( material.normalTexture );
	//		samplers.insert( normalTex.samplerIdx );
	//		metadata.normalIdx = normalTex.imageIdx;
	//
	//		const gltf_texture pbrBaseCol = ProcessTexture( pbrInfo.baseColorTexture );
	//		samplers.insert( pbrBaseCol.samplerIdx );
	//		metadata.baseColorIdx = pbrBaseCol.imageIdx;
	//
	//		const gltf_texture metallicRoughness = ProcessTexture( pbrInfo.metallicRoughnessTexture );
	//		samplers.insert( metallicRoughness.samplerIdx );
	//		metadata.metallicRoughnessIdx = metallicRoughness.imageIdx;
	//
	//		const gltf_texture occlusionTex = ProcessTexture( material.occlusionTexture );
	//		samplers.insert( occlusionTex.samplerIdx );
	//		metadata.occlusionIdx = occlusionTex.imageIdx;
	//
	//		const gltf_texture emissiveTex = ProcessTexture( material.emissiveTexture );
	//		samplers.insert( emissiveTex.samplerIdx );
	//		metadata.emissiveIdx = emissiveTex.imageIdx;
	//
	//		// NOTE: we will have -1 and possibly others so at most size 2
	//		HP_ASSERT( std::size( samplers ) <= 2 );
	//
	//		auto it = std::ranges::find_if( samplers,
	//										[]( i32 x ){ return x != -1; });
	//		metadata.samplerIdx = ( it != std::cend( samplers ) ) ? *it : -1;
	//
	//		materialsOut.emplace_back( metadata );
	//	}
	//
	//	return materialsOut;
	//}

	//std::vector<gltf_image> ProcessImages() const
	//{
	//	std::vector<gltf_image> imgOut;
	//	imgOut.reserve( std::size( model.images ) );
	//	for( const tinygltf::Image& img : model.images )
	//	{
	//		HP_ASSERT( std::size( img.image ) );
	//		imgOut.push_back( {
	//			.data = std::span<const u8>{ std::data( img.image ), std::size( img.image ) },
	//			.metadata = GetGltfTextureMetadata( img )
	//						  } );
	//	}
	//
	//	return imgOut;
	//}

	template<TinyGltfTextureInfoConcept TexInfo>
	inline gltf_texture ProcessTexture( const TexInfo& texInfo ) const
	{
		gltf_texture texOut = {};
		if( -1 != texInfo.index )
		{
			// NOTE: bc we use TEXCOORD_0
			HP_ASSERT( 0 == texInfo.texCoord );
			const tinygltf::Texture& tex = model.textures[ texInfo.index ];
			texOut = {
				.imageIdx = tex.source,
				.samplerIdx = tex.sampler
			};
		}

		return texOut;
	}
	// UTILS:
	static inline packed_trs GetTrsFromNode( const tinygltf::Node& node )
	{
		using namespace DirectX;

		XMVECTOR xmT = XMVectorSet( 0, 0, 0, 0 );
		XMVECTOR xmR = XMVectorSet( 0, 0, 0, 1 );
		XMVECTOR xmS = XMVectorSet( 1, 1, 1, 0 );
		if( std::size( node.matrix ) == 16 )
		{
			XMMATRIX m = GetMatrix( node.matrix );
			if( !XMMatrixDecompose( &xmS, &xmR, &xmT, m ) )
			{
				xmT = XMVectorSet( 0, 0, 0, 0 );
				xmR = XMVectorSet( 0, 0, 0, 1 );
				xmS = XMVectorSet( 1, 1, 1, 0 );
			}

		}
		else
		{
			if( std::size( node.translation ) == 3 )
			{
				xmT = XMVectorSet(
					(float)node.translation[0],
					(float)node.translation[1],
					(float)node.translation[2],
					0.0f
				);
			}
			if( std::size( node.rotation ) == 4 )
			{
				xmR = XMVectorSet(
					(float)node.rotation[0],
					(float)node.rotation[1],
					(float)node.rotation[2],
					(float)node.rotation[3]
				);
			}
			if( std::size( node.scale ) == 3 )
			{
				xmS = XMVectorSet(
					(float)node.scale[0],
					(float)node.scale[1],
					(float)node.scale[2],
					0.0f
				);
			}
		}

		float3 t;
		float4 r;
		float3 s;

		XMStoreFloat3( &t, xmT );
		XMStoreFloat4( &r, xmR );
		XMStoreFloat3( &s, xmS );
		return { .t = t, .r = r, .s = s };
	}
	// NOTE: gltf matrices are col maj, right-handed
	static inline DirectX::XMMATRIX GetMatrix( const std::vector<double>& mIn )
	{
		using namespace DirectX;
		XMMATRIX m = XMMatrixSet(
			( float ) mIn[ 0 ], ( float ) mIn[ 1 ], ( float ) mIn[ 2 ], ( float ) mIn[ 3 ],
			( float ) mIn[ 4 ], ( float ) mIn[ 5 ], ( float ) mIn[ 6 ], ( float ) mIn[ 7 ],
			( float ) mIn[ 8 ], ( float ) mIn[ 9 ], ( float ) mIn[ 10 ], ( float ) mIn[ 11 ],
			( float ) mIn[ 12 ], ( float ) mIn[ 13 ], ( float ) mIn[ 14 ], ( float ) mIn[ 15 ]
		);

		return XMMatrixTranspose( m );
	}

	inline const tinygltf::Accessor* GetAccessorByName( std::string_view name, const tinygltf::Primitive& primMesh ) const
	{
		const auto it = primMesh.attributes.find( std::string{ name } );
		if( std::cend( primMesh.attributes ) == it ) return nullptr;
		return &( model.accessors[ it->second ] );
	}

	// NOTE: according to the spec https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#meshes
	// we could just fix these attribute's size
	gltf_raw_attr_stream GetRawAttributeStream( const tinygltf::Accessor& accessor ) const
	{
		const i32 viewIdx = accessor.bufferView;
		HP_ASSERT( -1 != viewIdx );
		const tinygltf::BufferView& view = model.bufferViews[ viewIdx ];

		HP_ASSERT( -1 != view.buffer );
		const tinygltf::Buffer& buff = model.buffers[ view.buffer ];

		const u64 numComponents = tinygltf::GetNumComponentsInType( accessor.type );
		const u64 componentSizeInBytes = tinygltf::GetComponentSizeInBytes( accessor.componentType );
		const u64 elemSize = numComponents * componentSizeInBytes;

		// NOTE: gltf byteStride==0 means tightly packed ( effective stride == elemSize )
		const u64 strideInBytes = ( view.byteStride == 0 ) ? elemSize : ( u64 ) view.byteStride;

		// NOTE: If a stride is provided, it must be >= element size
		HP_ASSERT( strideInBytes >= elemSize );

		const u64 baseOffset = ( u64 ) view.byteOffset + ( u64 ) accessor.byteOffset;
		HP_ASSERT( baseOffset < ( u64 ) std::size( buff.data ) );

		const u8* streamView = ( const u8* ) ( std::data( buff.data ) + baseOffset );

		//// NOTE: avoid overflow, last element starts at (count-1)*stride, needs elemSize bytes
		//if( accessor.count > 0 )
		//{
		//	const u64 last = baseOffset + ( u64 ) ( accessor.count - 1 ) * stride + elemSize;
		//	HP_ASSERT( last <= ( u64 ) std::size( buff.data ) );
		//}

		return { streamView, accessor.count, numComponents, componentSizeInBytes, strideInBytes };
	}

	auto GetIndexBufferFromStream( const tinygltf::Accessor& idxAccessor ) const
	{
		HP_ASSERT( TINYGLTF_TYPE_SCALAR == idxAccessor.type );

		gltf_raw_attr_stream rawIdxStream = GetRawAttributeStream( idxAccessor );

		HP_ASSERT( ( std::size( rawIdxStream ) % 3 ) == 0 );

		std::vector<u32> normalized( std::size( rawIdxStream ) );

		auto CasterLambda = [] ( auto v ) { return ( u32 ) v; }; 

		switch( idxAccessor.componentType )
		{
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
		{
			gltf_typed_attr_stream<u8> typedIdxStream = { rawIdxStream };
			std::ranges::copy( typedIdxStream | std::views::transform( CasterLambda ), std::begin( normalized ) );
			break;
		}
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
		{
			gltf_typed_attr_stream<u16> typedIdxStream = { rawIdxStream };
			std::ranges::copy( typedIdxStream | std::views::transform( CasterLambda ), std::begin( normalized ) );
			break;
		}
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
		{
			gltf_typed_attr_stream<u32> typedIdxStream = { rawIdxStream };
			std::ranges::copy( typedIdxStream, std::begin( normalized ) );
			break;
		}
		}
		return normalized;
	}

	static inline alpha_mode GltfAlphaModeToEnum( std::string_view gltfAlphaMode )
	{
		if( gltfAlphaMode == "MASK" )  return ALPHA_MODE_MASK;
		if( gltfAlphaMode == "BLEND" ) return ALPHA_MODE_BLEND;
		return ALPHA_MODE_OPAQUE;
	}

	static inline sampler_filter_mode_flags GltfFilterToFlags( int gltfFilter )
	{
		switch( gltfFilter )
		{
		case TINYGLTF_TEXTURE_FILTER_NEAREST:                 return FILTER_NEAREST;
		case TINYGLTF_TEXTURE_FILTER_LINEAR:                  return FILTER_LINEAR;
		case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_NEAREST: return FILTER_NEAREST_MIPMAP_NEAREST;
		case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_NEAREST:  return FILTER_LINEAR_MIPMAP_NEAREST;
		case TINYGLTF_TEXTURE_FILTER_NEAREST_MIPMAP_LINEAR:  return FILTER_NEAREST_MIPMAP_LINEAR;
		case TINYGLTF_TEXTURE_FILTER_LINEAR_MIPMAP_LINEAR:   return FILTER_LINEAR_MIPMAP_LINEAR;
		default:                                             return FILTER_LINEAR;
		}
	}

	static inline sampler_wrap_mode_flags GltfWrapToFlags( int gltfWrap )
	{
		switch( gltfWrap )
		{
		case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:   return WRAP_CLAMP_TO_EDGE;
		case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT: return WRAP_MIRRORED_REPEAT;
		case TINYGLTF_TEXTURE_WRAP_REPEAT:          return WRAP_REPEAT;
		default:                                    return WRAP_CLAMP_TO_EDGE;
		}
	}

	static inline gltf_image_metadata GetGltfTextureMetadata( const tinygltf::Image& img )
	{
		gltf_image_metadata out{
			.width = img.width,
			.height = img.height,
			//.component = ,
			//.bits = ,
			//.pixelType =
		};

		switch( img.component )
		{
		case 1: out.component = gltf_img_components::R; break;
		case 2: out.component = gltf_img_components::RG; break;
		case 3: out.component = gltf_img_components::RGB; break;
		case 4: out.component = gltf_img_components::RGBA; break;
		default: out.component = gltf_img_components::UNKNOWN;
		}

		switch( img.bits )
		{
		case 8:  out.bits = gltf_img_bit_depth::B8; break;
		case 16: out.bits = gltf_img_bit_depth::B16; break;
		case 32: out.bits = gltf_img_bit_depth::B32; break;
		default: out.bits = gltf_img_bit_depth::UNKNOWN;
		}

		switch( img.pixel_type )
		{
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:  out.pixelType = gltf_img_pixel_type::UBYTE; break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT: out.pixelType = gltf_img_pixel_type::USHORT; break;
		case TINYGLTF_COMPONENT_TYPE_FLOAT:          out.pixelType = gltf_img_pixel_type::FLOAT32; break;
		default: out.pixelType = gltf_img_pixel_type::UNKNOWN;
		}

		return out;
	}
};


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

__meshopt_meshlets MakeMeshletsRaytracing( 
	std::span<const float3> pos, 
	std::span<const u32> indices, 
	cluster_raytracing clusterConfig 
) {
	const u64 indexCount = std::size( indices );

	// NOTE ( meshoptimizer ): use minTriangles to compute worst case bound
	const u64 maxMeshletCount = meshopt_buildMeshletsBound( indexCount, clusterConfig.maxVertices, clusterConfig.minTriangles );
	std::vector<meshopt_Meshlet> meshlets( maxMeshletCount );
	std::vector<u32> mletVtx( indexCount );
	std::vector<u8> mletTris( indexCount );

	const u64 meshletCount = meshopt_buildMeshletsSpatial( 
		std::data( meshlets ), std::data( mletVtx ), std::data( mletTris ), std::data( indices ), std::size( indices ), 
		&pos[ 0 ].x, std::size( pos ), sizeof( pos[ 0 ] ), clusterConfig.maxVertices, 
		clusterConfig.minTriangles, clusterConfig.maxTriangles, clusterConfig.fillWeight);

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

auto PackMeshletsRaytracing( const raw_mesh& rawMesh )
{
	__meshopt_meshlets meshlets = MakeMeshletsRaytracing( rawMesh.pos, rawMesh.indices, cluster_raytracing{} );

	std::span<const float3> pos = rawMesh.pos;
	std::span<const float3> norm = rawMesh.normals;
	std::span<const float4> tan = rawMesh.tans;
	std::span<const float2> uvs = rawMesh.uvs;

	std::vector<rt_meshlet> packedMeshlets;
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

		std::vector<packed_vtx> packedAttrs( std::size( mletNormStream ) );
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

		rt_meshlet rtMeshlet = {
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


struct instance
{
	packed_trs toWorld;
	bvh2_node_ref32 clasBvhRoot;
	u32 clasNodeCount;
	u32 baseMeshletOffset;
	u32 meshletCount;
	i32 materialIdx;
};

struct world_data
{
	std::vector<instance> instances;
	std::vector<gpu_bvh2_node> globalTlasBuffer;
	std::vector<gpu_bvh2_node> globalClasBuffer;
	std::vector<rt_meshlet_info> meshletInfoBuffer;
	std::vector<float3> globalVertexPosBuffer;
	std::vector<packed_vtx> globalPackedVertexBuffer;
	std::vector<u8> globalTriangleBuffer;

	range64 AppendMeshlets( const std::ranges::forward_range auto& meshlets )
	{
		HP_ASSERT( std::size( globalVertexPosBuffer ) == std::size( globalPackedVertexBuffer ) );
		HP_ASSERT( std::size( globalVertexPosBuffer ) < u32( -1 ) );
		HP_ASSERT( std::size( globalTriangleBuffer ) < u32( -1 ) );
		HP_ASSERT( std::size( globalClasBuffer ) < u32( -1 ) );

		u64 totalVertexCount = 0;
		u64 totalTrianlgeCount = 0;
		for( const rt_meshlet& m : meshlets )
		{
			const u64 vertexCount = std::size( m.positions );
			const u64 triangleCount = std::size( m.triangles );

			HP_ASSERT( vertexCount == std::size( m.packedAttrs ) );
			HP_ASSERT( ( vertexCount <= u8( -1 ) ) && ( triangleCount <= u8( -1 ) ) );

			totalVertexCount += vertexCount;
			totalTrianlgeCount += triangleCount;
		}
		globalVertexPosBuffer.reserve( std::size( globalVertexPosBuffer ) + totalVertexCount );
		globalPackedVertexBuffer.reserve( std::size( globalPackedVertexBuffer ) + totalVertexCount );
		globalTriangleBuffer.reserve( std::size( globalTriangleBuffer ) + totalTrianlgeCount );

		HP_ASSERT( std::size( meshletInfoBuffer ) <= u32( -1 ) );
		const u32 baseMeshletOffset = ( u32 ) std::size( meshletInfoBuffer );
		meshletInfoBuffer.reserve( baseMeshletOffset + std::size( meshlets ) );

		for( const rt_meshlet& m : meshlets )
		{
			rt_meshlet_info info = {
				.aabbMin = m.aabbMin,
				.aabbMax = m.aabbMax,
				.vertexOffset = ( u32 ) std::size( globalVertexPosBuffer ),
				.triangleOffset = ( u32 ) std::size( globalTriangleBuffer ),
				.vertexCount = ( u8 ) std::size( m.positions ),
				.triangleCount = ( u8 ) std::size( m.triangles )
			};

			std::ranges::copy( m.positions, std::back_inserter( globalVertexPosBuffer ) );
			std::ranges::copy( m.packedAttrs, std::back_inserter( globalPackedVertexBuffer ) );
			std::ranges::copy( m.triangles, std::back_inserter( globalTriangleBuffer ) );
			meshletInfoBuffer.push_back( info );
		}

		HP_ASSERT( std::size( meshlets ) <= u32( -1 ) );
		return { .baseOffset = baseMeshletOffset, .count = ( u32 ) std::size( meshlets ) };
	}
};

template<typename T, typename Idx>
inline auto PermutedView( std::vector<T>& src, const std::vector<Idx>& remap )
{
	return remap | std::views::transform( [&]( Idx oldIdx ) -> T& { return src[ oldIdx ]; } );
}

template<typename T, typename Idx>
inline auto PermutedView( const std::vector<T>& src, const std::vector<Idx>& remap )
{
	return remap | std::views::transform( [&]( Idx oldIdx ) -> const T& { return src[ oldIdx ]; } );
}

inline auto MeshletAabbView( const std::ranges::forward_range auto& meshlets )
{
	return meshlets | std::views::transform( 
		[] ( const rt_meshlet& m ) { return aabb_t<float3>{ .min = m.aabbMin, .max = m.aabbMax }; } );
};

int main()
{
	const std::string gltfFilePath = "D:/3d models/nightclub_futuristic_pub_ambience_asset.glb";
	HP_ASSERT( fs::exists( gltfFilePath ) );

	gltf_processor gltf = { gltfFilePath };

	// TODO: ensure we keep the same indexing !!!!
	std::vector<raw_node> rawNodes = gltf.ProcessNodes();
	std::vector<raw_mesh> rawMeshes = gltf.ProcessMeshes();

	world_data worldData = {};

	bvh_builder bvhBuilder = {};

	std::vector<aabb_t<float3>> tlasAabbs;
	tlasAabbs.reserve( std::size( rawMeshes ) );

	for( const raw_node& n : rawNodes )
	{
		if( INVALID_IDX == n.meshIdx ) continue;

		raw_mesh& rawMesh = rawMeshes[ n.meshIdx ];
		if( !std::size( rawMesh.tans ) )
		{
			ComputeMikkTSpaceTangentsInplace( rawMesh );
		}

		std::vector<rt_meshlet> packedMeshlets = PackMeshletsRaytracing( rawMesh );
		HP_ASSERT( std::size( packedMeshlets ) != 0 );

		auto meshletAabbView = MeshletAabbView( packedMeshlets );
		// NOTE: if we don't have at least 3 leaves worth of work we can skip the BVH
		const bool buildBvhThresholdReached = std::size( meshletAabbView ) <= 2 * MAX_LEAF_PRIM_COUNT;
		if( buildBvhThresholdReached )
		{
			auto[ baseMeshletOffset, meshletCount ] = worldData.AppendMeshlets( packedMeshlets );

			aabb_t<float3> tlasAabb = MergeAabbs( meshletAabbView );
			tlasAabbs.push_back( tlasAabb );

			worldData.instances.push_back( {
				.toWorld = n.toWorld,
				.clasBvhRoot = BVH_INVALID_REF,
				.clasNodeCount = 0,
				.baseMeshletOffset = ( u32 ) baseMeshletOffset,
				.meshletCount = ( u32 ) meshletCount,
				.materialIdx = rawMesh.materialIdx
			} );
		}
		else
		{
			bvh_output clasOutput = bvhBuilder.BuildBvhOverPrimitives( meshletAabbView );

			const u32 bvhRootOffset = ( u32 ) std::size( worldData.globalClasBuffer );
			const u32 clasNodeCount = ( u32 ) std::size( clasOutput.gpuNodes );
			worldData.globalClasBuffer.reserve( bvhRootOffset + clasNodeCount ); // NOTE: just convenice to not type size( ... )
			std::ranges::copy( clasOutput.gpuNodes, std::back_inserter( worldData.globalClasBuffer ) );

			auto permutedView = PermutedView( packedMeshlets, clasOutput.primitiveIndices );
			auto[ baseMeshletOffset, meshletCount ] = worldData.AppendMeshlets( permutedView );

			tlasAabbs.push_back( clasOutput.topLevelAabb );

			worldData.instances.push_back( {
				.toWorld = n.toWorld,
				.clasBvhRoot = bvhRootOffset,
				.clasNodeCount = clasNodeCount,
				.baseMeshletOffset = ( u32 ) baseMeshletOffset,
				.meshletCount = ( u32 ) meshletCount,
				.materialIdx = rawMesh.materialIdx
			} );
		}
	}

	bvh_output tlasOutput = bvhBuilder.BuildBvhOverPrimitives( tlasAabbs );
	worldData.globalTlasBuffer = std::move( tlasOutput.gpuNodes );
}

