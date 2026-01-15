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

/**
 * Temporal accumulation render pass.
 *
 * There are entry points for each of the three supported accumulation modes.
 * Note that for the compensated summation mode the shader _must_ be compiled
 * for precise floating-point operations (no reordering).
 *
 * In all modes, the shader writes the current accumulated average to the
 * output texture. The intermediate buffers are internal to the pass.
 */

cbuffer PerFrameCB
{
    uint2 gResolution;
    uint gAccumCount;
    bool gAccumulate;
    bool gMovingAverageMode;
}

Texture2D<float4> gColor;
Texture2D<float>  gBounceMask;
Texture2D<float4> gBouncePosW;
Texture2D<float4> gBounceNormalW;
Texture2D<float4> gBounceDirW;
Texture2D<float4> gBounceDiffRough;
Texture2D<float4> gBounceEmissive;

RWTexture2D<float4> gOutColor;
RWTexture2D<float>  gOutBounceMask;
RWTexture2D<float4> gOutBouncePosW;
RWTexture2D<float4> gOutBounceNormalW;
RWTexture2D<float4> gOutBounceDirW;
RWTexture2D<float4> gOutBounceDiffRough;
RWTexture2D<float4> gOutBounceEmissive;

// Last frame data, format depends on accumulation mode.
RWTexture2D<float4> gLastFrameSumColor;
RWTexture2D<float>  gLastFrameSumBounceMask;
RWTexture2D<float4> gLastFrameSumBouncePosW;
RWTexture2D<float4> gLastFrameSumBounceNormalW;
RWTexture2D<float4> gLastFrameSumBounceDirW;
RWTexture2D<float4> gLastFrameSumBounceDiffRough;
RWTexture2D<float4> gLastFrameSumBounceEmissive;

void accumulate_func(float curVal, inout float lastFrameSum, inout float result, bool accumulate)
{
    if (accumulate) 
    {
        float curWeight = 1.0 / (gAccumCount + 1);
        if (gMovingAverageMode)
        {
            // Exponential weighted moving average mode.
            // Blend the accumulated frame with the current frame.
            result = lerp(lastFrameSum, curVal, curWeight);
            lastFrameSum = result;
        }
        else
        {
            // High precision mode with no frame limit.
            // Fetch previous sum and compute the new sum.
            float sum = lastFrameSum + curVal;
            result = sum * curWeight;
            lastFrameSum = sum;
        }
    }
    else
    {
        result = curVal;
    }
}

void accumulate_func(float4 curVal, inout float4 lastFrameSum, inout float4 result, bool accumulate)
{
    if (accumulate) 
    {
        float curWeight = 1.0 / (gAccumCount + 1);
        if (gMovingAverageMode)
        {
            // Exponential weighted moving average mode.
            // Blend the accumulated frame with the current frame.
            result = lerp(lastFrameSum, curVal, curWeight);
            lastFrameSum = result;
        }
        else
        {
            // High precision mode with no frame limit.
            // Fetch previous sum and compute the new sum.
            float4 sum = lastFrameSum + curVal;
            result = sum * curWeight;
            lastFrameSum = sum;
        }
    }
    else
    {
        result = curVal;
    }
}

#define handle(feat) \
    curVal = g##feat[inputPixelPos]; \
    lastFrameSum = gLastFrameSum##feat[outputPixelPos]; \
    accumulate_func(curVal, lastFrameSum, result, gAccumulate); \
    gOut##feat[outputPixelPos] = result; \
    gLastFrameSum##feat[outputPixelPos] = lastFrameSum

#define handleFloat(feat) \
    curValFloat = g##feat[inputPixelPos]; \
    lastFrameSumFloat = gLastFrameSum##feat[outputPixelPos]; \
    accumulate_func(curValFloat, lastFrameSumFloat, resultFloat, gAccumulate); \
    gOut##feat[outputPixelPos] = resultFloat; \
    gLastFrameSum##feat[outputPixelPos] = lastFrameSumFloat

/**
 * Single precision standard summation.
 */
[numthreads(16, 16, 1)]
void accumulateSingle(uint3 dispatchThreadId: SV_DispatchThreadID)
{    
    if (any(dispatchThreadId.xy >= gResolution))
        return;

    const uint2 outputPixelPos = dispatchThreadId.xy;
    const uint2 inputPixelPos = dispatchThreadId.xy;

    float4 curVal;
    float4 lastFrameSum;
    float4 result;
    float curValFloat;
    float lastFrameSumFloat;
    float resultFloat;

    handle(Color);

    handleFloat(BounceMask);
    handle(BouncePosW);
    handle(BounceNormalW);
    handle(BounceDirW);
    handle(BounceDiffRough);
    handle(BounceEmissive);
}
