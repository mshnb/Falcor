/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "GBuffer.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Rendering/Materials/TexLODTypes.slang"

using namespace Falcor;

/**
 * Refractive Ray traced G-buffer pass.
 * This pass renders a fixed set of G-buffer channels using ray tracing.
 */
class MixedGBufferRT : public GBuffer
{
public:
    FALCOR_PLUGIN_CLASS(MixedGBufferRT, "MixedGBufferRT", "Refractive Ray traced G-buffer generation pass.");

    static ref<MixedGBufferRT> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<MixedGBufferRT>(pDevice, props);
    }

    MixedGBufferRT(ref<Device> pDevice, const Properties& props);

    RenderPassReflection reflect(const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    Properties getProperties() const override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;

private:
    void parseProperties(const Properties& props) override;

    void executeRaytrace(RenderContext* pRenderContext, const RenderData& renderData);
    void executeCompute(RenderContext* pRenderContext, const RenderData& renderData);

    DefineList getShaderDefines(const RenderData& renderData) const;
    void bindShaderData(const ShaderVar& var, const RenderData& renderData);
    void recreatePrograms();

    // Internal state

    /// Flag indicating if depth-of-field is computed for the current frame.
    bool mComputeDOF = false;
    ref<SampleGenerator> mpSampleGenerator;

    // UI variables

    TexLODMode mLODMode = TexLODMode::Mip0;
    bool mUseTraceRayInline = false;
    /// Option for enabling depth-of-field when camera's aperture radius is nonzero.
    bool mUseDOF = true;

    // Ray tracing resources
    struct
    {
        ref<Program> pProgram;
        ref<RtProgramVars> pVars;
    } mRaytrace;

    ref<ComputePass> mpComputePass;
    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;
    uint32_t mMaxBounces = 3;
};
