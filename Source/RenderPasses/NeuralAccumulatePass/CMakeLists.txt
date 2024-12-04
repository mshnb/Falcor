add_plugin(NeuralAccumulatePass)

target_sources(NeuralAccumulatePass PRIVATE
    NeuralAccumulatePass.cs.slang
    NeuralAccumulatePass.cpp
    NeuralAccumulatePass.h
)

target_copy_shaders(NeuralAccumulatePass RenderPasses/NeuralAccumulatePass)
target_source_group(NeuralAccumulatePass "RenderPasses")
