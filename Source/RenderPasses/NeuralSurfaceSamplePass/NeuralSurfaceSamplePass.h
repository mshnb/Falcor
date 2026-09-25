#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
using namespace Falcor;

class NeuralSurfaceSamplePass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(NeuralSurfaceSamplePass, "NeuralSurfaceSamplePass", "Stratified first-hit surface sampling for neural training data.");
    static ref<NeuralSurfaceSamplePass> create(ref<Device> device, const Properties& props) { return make_ref<NeuralSurfaceSamplePass>(device, props); }
    NeuralSurfaceSamplePass(ref<Device> device, const Properties& props);
    Properties getProperties() const override;
    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* context, const RenderData& data) override;
    void renderUI(Gui::Widgets& widget) override {}
private:
    ref<ComputePass> mpCompute;
    uint32_t mSampleCount = 8192;
    uint32_t mSeed = 0;
};
