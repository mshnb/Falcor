add_plugin(LightSample)

target_sources(LightSample PRIVATE
    LightSample.cs.slang
    LightSample.cpp
    LightSample.h
)

target_copy_shaders(LightSample RenderPasses/LightSample)
target_source_group(LightSample "RenderPasses")
