add_plugin(GBuffer)

target_sources(GBuffer PRIVATE
    GBufferBase.cpp
    GBufferBase.h

    GBuffer/DepthPass.3d.slang
    GBuffer/GBuffer.cpp
    GBuffer/GBuffer.h
    GBuffer/GBufferHelpers.slang
    GBuffer/GBufferRaster.3d.slang
    GBuffer/GBufferRaster.cpp
    GBuffer/GBufferRaster.h
    GBuffer/GBufferRT.cpp
    GBuffer/GBufferRT.cs.slang
    GBuffer/GBufferRT.h
    GBuffer/GBufferRT.rt.slang
    GBuffer/GBufferRT.slang

    GBuffer/ReflectGBufferRT.cpp
    GBuffer/ReflectGBufferRT.h
    GBuffer/ReflectGBufferRT.rt.slang
    GBuffer/ReflectGBufferRT.slang

    GBuffer/RefractGBufferRT.cpp
    GBuffer/RefractGBufferRT.h
    GBuffer/RefractGBufferRT.rt.slang
    GBuffer/RefractGBufferRT.slang

    GBuffer/MixedGBufferRT.cpp
    GBuffer/MixedGBufferRT.h
    GBuffer/MixedGBufferRT.rt.slang
    GBuffer/MixedGBufferRT.slang

    VBuffer/VBufferRaster.3d.slang
    VBuffer/VBufferRaster.cpp
    VBuffer/VBufferRaster.h
    VBuffer/VBufferRT.cpp
    VBuffer/VBufferRT.cs.slang
    VBuffer/VBufferRT.h
    VBuffer/VBufferRT.rt.slang
    VBuffer/VBufferRT.slang
)

target_copy_shaders(GBuffer RenderPasses/GBuffer)

target_source_group(GBuffer "RenderPasses")
