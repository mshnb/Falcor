#pragma once
#include "Window.h"
#include "Core/Object.h"
#include "Core/API/Device.h"
#include "Core/API/Swapchain.h"
#include "Core/API/Formats.h"
#include "RenderGraph/RenderGraph.h"
#include "Scene/Scene.h"
#include "Scene/SceneBuilder.h"
#include "Utils/Scripting/ndarray.h"
#include "Utils/Image/ImageProcessing.h"
#include "Utils/Timing/FrameRate.h"
#include "Utils/UI/PythonUI.h"
#include "Utils/Timing/Clock.h"
#include <memory>
#include <filesystem>

namespace Falcor
{

class ProfilerUI;

class Sandbox : public Object, private Window::ICallbacks
{
    FALCOR_OBJECT(Sandbox)
public:
    struct Options
    {
        // Note: Empty constructor needed for clang due to the use of the nested struct constructor in the parent constructor.
        Options() {}
        ref<Device> pDevice = nullptr;
        /// Desc for creating a new device if an existing device isn't already available.
        Device::Desc deviceDesc;
        Window::Desc windowDesc;
        bool createWindow = false;

        /// Color format of the frame buffer.
        ResourceFormat colorFormat = ResourceFormat::BGRA8UnormSrgb;
        /// Depth buffer format of the frame buffer.
        ResourceFormat depthFormat = ResourceFormat::D32Float;
    };

    static ref<Sandbox> create(const Options& options) { return make_ref<Sandbox>(options); }

    Sandbox(const Options& options = Options());
    virtual ~Sandbox();

    const ref<Device>& getDevice() const { return mpDevice; }

    /// Run the main loop.
    /// This only returns if the application window is closed or the main loop is interrupted by calling interrupt().
    void run(uint32_t spp);

    /// Interrupt the main loop.
    void interrupt();

    /// Close the testbed.
    /// This results in `shouldClose` to return true on the next call.
    void close();

    /// Render a single frame.
    /// Note: This is called repeatadly when running the main loop.
    void frame();

    /// Resize the main frame buffer.
    void resizeFrameBuffer(uint32_t width, uint32_t height);

    /// Load a scene.
    void loadScene(const std::filesystem::path& path, SceneBuilder::Flags buildFlags = SceneBuilder::Flags::Default);

    /// Load a scene from a string.
    void loadSceneFromString(
        const std::string& sceneStr,
        const std::string extension = "pyscene",
        SceneBuilder::Flags buildFlags = SceneBuilder::Flags::Default
    );

    ref<Scene> getScene() const;
    Clock& getClock();

    /// Create a new render graph.
    ref<RenderGraph> createRenderGraph(const std::string& name = "");

    /// Load a render graph from a file.
    ref<RenderGraph> loadRenderGraph(const std::filesystem::path& path);

    /// Set the active render graph.
    void setRenderGraph(const ref<RenderGraph>& graph);

    /// Get the active render graph.
    const ref<RenderGraph>& getRenderGraph() const;

    /// Get the python UI screen.
    const ref<python_ui::Screen>& getScreen() const { return mpScreen; }

    /// Capture the output of a render graph and write it to a file.
    void captureOutput(const std::filesystem::path& path, uint32_t outputIndex = 0);

    /// Set the texture to be rendered on the main window (overrides render graph output).
    void setRenderTexture(const ref<Texture>& texture) { mpRenderTexture = texture; }

    /// Get the texture to be rendered on the main window.
    const ref<Texture>& getRenderTexture() const { return mpRenderTexture; }

    /// Set the UI visibility.
    void setShowUI(bool showUI) { mUI.showUI = showUI; }

    /// Get the UI visibility.
    bool getShowUI() const { return mUI.showUI; }

    /// Returns true if the application should terminate.
    /// This is true if the window was closed or escape was pressed.
    bool shouldClose() const { return mShouldClose || (mpWindow && mpWindow->shouldClose()); }

    std::map<std::string, pybind11::ndarray<pybind11::pytorch>> getOutputsTorch() const;

private:
    // Implementation of Window::ICallbacks

    void handleWindowSizeChange() override;
    void handleRenderFrame() override;
    void handleKeyboardEvent(const KeyboardEvent& keyEvent) override;
    void handleMouseEvent(const MouseEvent& mouseEvent) override;
    void handleGamepadEvent(const GamepadEvent& gamepadEvent) override;
    void handleGamepadState(const GamepadState& gamepadState) override;
    void handleDroppedFile(const std::filesystem::path& path) override;

    void internalInit(const Options& options);
    void internalShutdown();

    void resizeTargetFBO(uint32_t width, uint32_t height);

    void renderUI();

    ref<Device> mpDevice;
    ref<Window> mpWindow;
    ref<Swapchain> mpSwapchain;
    ref<Fbo> mpTargetFBO;
    std::unique_ptr<Gui> mpGui;
    ref<python_ui::Screen> mpScreen;
    std::unique_ptr<ProfilerUI> mpProfilerUI;

    ref<Scene> mpScene;
    ref<RenderGraph> mpRenderGraph;
    ref<Texture> mpRenderTexture;

    std::unique_ptr<ImageProcessing> mpImageProcessing;

    FrameRate mFrameRate;
    Clock mClock;

    bool mShouldInterrupt{false};
    bool mShouldClose{false};
    struct
    {
        bool showUI = true;
        bool showFPS = true;
    } mUI;
};

} // namespace Falcor
