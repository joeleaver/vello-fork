// Copyright 2022 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Vello is a compute-centric 2D graphics rendering engine written in Rust,
//! using [`wgpu`].
//! It efficiently draws large 2D scenes with interactive or near-interactive
//! performance.
//!
//! ![image](https://github.com/linebender/vello/assets/8573618/cc2b742e-2135-4b70-8051-c49aeddb5d19)
//!
//!
//! ## Motivation
//!
//! Vello is meant to fill the same place in the graphics stack as other vector graphics renderers like [Skia](https://skia.org/), [Cairo](https://www.cairographics.org/), and its predecessor project [Piet](https://www.cairographics.org/).
//! On a basic level, that means it provides tools to render shapes, images, gradients, texts, etc, using a PostScript-inspired API, the same that powers SVG files and [the browser `<canvas>` element](https://developer.mozilla.org/en-US/docs/Web/API/CanvasRenderingContext2D).
//!
//! Vello's selling point is that it gets better performance than other renderers by better leveraging the GPU.
//! In traditional PostScript-style renderers, some steps of the render process like sorting and clipping either need to be handled in the CPU or done through the use of intermediary textures.
//! Vello avoids this by using prefix-scan algorithms to parallelize work that usually needs to happen in sequence, so that work can be offloaded to the GPU with minimal use of temporary buffers.
//!
//! This means that Vello needs a GPU with support for compute shaders to run.
//!
//!
//! ## Getting started
//!
//! Vello is meant to be integrated deep in UI render stacks.
//! While drawing in a Vello [`Scene`] is easy, actually rendering that scene to a surface setting up a wgpu context, which is a non-trivial task.
//!
//! To use Vello as the renderer for your PDF reader / GUI toolkit / etc, your code will have to look roughly like this:
//!
//! ```ignore
//! // Initialize wgpu and get handles
//! let (width, height) = ...;
//! let device: wgpu::Device = ...;
//! let queue: wgpu::Queue = ...;
//! let mut renderer = Renderer::new(
//!    &device,
//!    RendererOptions {
//!       use_cpu: false,
//!       antialiasing_support: vello::AaSupport::all(),
//!       num_init_threads: NonZeroUsize::new(1),
//!    },
//! ).expect("Failed to create renderer");
//!
//! // Create scene and draw stuff in it
//! let mut scene = vello::Scene::new();
//! scene.fill(
//!    vello::peniko::Fill::NonZero,
//!    vello::Affine::IDENTITY,
//!    vello::Color::from_rgb8(242, 140, 168),
//!    None,
//!    &vello::Circle::new((420.0, 200.0), 120.0),
//! );
//!
//! // Draw more stuff
//! scene.push_layer(...);
//! scene.fill(...);
//! scene.stroke(...);
//! scene.pop_layer(...);
//!
//! let texture = device.create_texture(&...);
//! // Render to a wgpu Texture
//! renderer
//!    .render_to_texture(
//!       &device,
//!       &queue,
//!       &scene,
//!       &texture,
//!       &vello::RenderParams {
//!          base_color: palette::css::BLACK, // Background color
//!          width,
//!          height,
//!          antialiasing_method: AaConfig::Msaa16,
//!       },
//!    )
//!    .expect("Failed to render to a texture");
//! // Do things with surface texture, such as blitting it to the Surface using
//! // wgpu::util::TextureBlitter.
//! ```
//!
//! See the [`research/examples/`](https://github.com/linebender/vello/tree/main/research/examples) folder to see how that code integrates with frameworks like winit.

// LINEBENDER LINT SET - lib.rs - v2
// See https://linebender.org/wiki/canonical-lints/
// These lints aren't included in Cargo.toml because they
// shouldn't apply to examples and tests
#![warn(unused_crate_dependencies)]
#![warn(clippy::print_stdout, clippy::print_stderr)]
// Targeting e.g. 32-bit means structs containing usize can give false positives for 64-bit.
#![cfg_attr(target_pointer_width = "64", warn(clippy::trivially_copy_pass_by_ref))]
// END LINEBENDER LINT SET
#![cfg_attr(docsrs, feature(doc_cfg))]
// The following lints are part of the Linebender standard set,
// but resolving them has been deferred for now.
// Feel free to send a PR that solves one or more of these.
// Need to allow instead of expect until Rust 1.83 https://github.com/rust-lang/rust/pull/130025
#![allow(missing_docs, reason = "We have many as-yet undocumented items.")]
#![expect(
    missing_debug_implementations,
    clippy::cast_possible_truncation,
    clippy::missing_assert_message,
    reason = "Deferred"
)]
#![allow(
    clippy::todo,
    unreachable_pub,
    unnameable_types,
    reason = "Deferred, only apply in some feature sets so not expect"
)]

// A dev-dependency used only by the bind-group cache's GPU tests, which need
// `wgpu` off WebAssembly: suppress `unused_crate_dependencies` elsewhere.
#[cfg(all(test, not(all(feature = "wgpu", not(target_arch = "wasm32")))))]
use pollster as _;

mod debug;
mod recording;
mod render;
mod scene;
mod shaders;

#[cfg(feature = "wgpu")]
mod bump_report;
#[cfg(feature = "wgpu")]
pub mod util;
#[cfg(feature = "wgpu")]
mod wgpu_engine;

#[cfg(feature = "wgpu")]
pub use bump_report::BumpReport;

pub mod low_level {
    //! Utilities which can be used to create an alternative Vello renderer to [`Renderer`][crate::Renderer].
    //!
    //! These APIs have not been carefully designed, and might not be powerful enough for this use case.

    pub use crate::debug::DebugLayers;
    pub use crate::recording::{
        BindType, BufferProxy, Command, ImageFormat, ImageProxy, Recording, ResourceId,
        ResourceProxy, ShaderId,
    };
    pub use crate::render::Render;
    pub use crate::shaders::FullShaders;

    pub use vello_encoding;
}
/// Styling and composition primitives.
pub use peniko;
/// 2D geometry, with a focus on curves.
pub use peniko::kurbo;

#[cfg(feature = "wgpu")]
use peniko::ImageData;
#[cfg(feature = "wgpu")]
pub use wgpu;

pub use scene::{DrawGlyphs, Scene};
pub use vello_encoding::{FontEmbolden, Glyph, NormalizedCoord};

use low_level::ShaderId;
#[cfg(feature = "wgpu")]
use low_level::{FullShaders, Recording, Render};
use thiserror::Error;

#[cfg(feature = "wgpu")]
use debug::DebugLayers;
#[cfg(feature = "wgpu")]
use vello_encoding::{BumpAllocators, ConfigUniform, Resolver};
#[cfg(feature = "wgpu")]
use wgpu_engine::{ExternalResource, WgpuEngine};

#[cfg(feature = "wgpu")]
use std::{
    collections::VecDeque,
    num::NonZeroUsize,
    sync::{Arc, OnceLock, atomic::AtomicBool},
};
#[cfg(feature = "wgpu")]
use wgpu::{Device, Queue, TextureView};
#[cfg(all(feature = "wgpu", feature = "wgpu-profiler"))]
use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};

/// Represents the anti-aliasing method to use during a render pass.
///
/// Can be configured for a render operation by setting [`RenderParams::antialiasing_method`].
/// Each value of this can only be used if the corresponding field on [`AaSupport`] was used.
///
/// This can be converted into an `AaSupport` using [`Iterator::collect`],
/// as `AaSupport` implements `FromIterator`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum AaConfig {
    /// Area anti-aliasing, where the alpha value for a pixel is computed from integrating
    /// the winding number over its square area.
    ///
    /// This technique produces very accurate values when the shape has winding number of 0 or 1
    /// everywhere, but can result in conflation artifacts otherwise.
    /// It generally has better performance than the multi-sampling methods.
    ///
    /// Can only be used if [enabled][AaSupport::area] for the `Renderer`.
    Area,
    /// 8x Multisampling
    ///
    /// Can only be used if [enabled][AaSupport::msaa8] for the `Renderer`.
    Msaa8,
    /// 16x Multisampling
    ///
    /// Can only be used if [enabled][AaSupport::msaa16] for the `Renderer`.
    Msaa16,
}

/// Represents the set of anti-aliasing configurations to enable during pipeline creation.
///
/// This is configured at `Renderer` creation time ([`Renderer::new`]) by setting
/// [`RendererOptions::antialiasing_support`].
///
/// This can be created from a set of `AaConfig` using [`Iterator::collect`],
/// as `AaSupport` implements `FromIterator`.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct AaSupport {
    /// Support [`AaConfig::Area`].
    pub area: bool,
    /// Support [`AaConfig::Msaa8`].
    pub msaa8: bool,
    /// Support [`AaConfig::Msaa16`].
    pub msaa16: bool,
}

impl AaSupport {
    /// Support every anti-aliasing method.
    ///
    /// This might increase startup time, as more shader variations must be compiled.
    pub fn all() -> Self {
        Self {
            area: true,
            msaa8: true,
            msaa16: true,
        }
    }

    /// Support only [`AaConfig::Area`].
    ///
    /// This should be the default choice for most users.
    pub fn area_only() -> Self {
        Self {
            area: true,
            msaa8: false,
            msaa16: false,
        }
    }
}

impl FromIterator<AaConfig> for AaSupport {
    fn from_iter<T: IntoIterator<Item = AaConfig>>(iter: T) -> Self {
        let mut result = Self {
            area: false,
            msaa8: false,
            msaa16: false,
        };
        for config in iter {
            match config {
                AaConfig::Area => result.area = true,
                AaConfig::Msaa8 => result.msaa8 = true,
                AaConfig::Msaa16 => result.msaa16 = true,
            }
        }
        result
    }
}

/// Errors that can occur in Vello.
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// There is no available device with the features required by Vello.
    #[cfg(feature = "wgpu")]
    #[error("Couldn't find suitable device")]
    NoCompatibleDevice,
    /// Failed to create surface.
    /// See [`wgpu::CreateSurfaceError`] for more information.
    #[cfg(feature = "wgpu")]
    #[error("Couldn't create wgpu surface")]
    WgpuCreateSurfaceError(#[from] wgpu::CreateSurfaceError),
    /// Surface doesn't support the required texture formats.
    /// Make sure that you have a surface which provides one of
    /// [`TextureFormat::Rgba8Unorm`][wgpu::TextureFormat::Rgba8Unorm]
    /// or [`TextureFormat::Bgra8Unorm`][wgpu::TextureFormat::Bgra8Unorm] as texture formats.
    // TODO: Why does this restriction exist?
    #[cfg(feature = "wgpu")]
    #[error("Couldn't find `Rgba8Unorm` or `Bgra8Unorm` texture formats for surface")]
    UnsupportedSurfaceFormat,

    /// Used a buffer inside a recording while it was not available.
    /// Check if you have created it and not freed before its last usage.
    #[cfg(feature = "wgpu")]
    #[error("Buffer '{0}' is not available but used for {1}")]
    UnavailableBufferUsed(&'static str, &'static str),
    /// Failed to async map a buffer.
    /// See [`wgpu::BufferAsyncError`] for more information.
    #[cfg(feature = "wgpu")]
    #[error("Failed to async map a buffer")]
    BufferAsyncError(#[from] wgpu::BufferAsyncError),
    /// Failed to download an internal buffer for debug visualization.
    #[cfg(feature = "wgpu")]
    #[cfg(feature = "debug_layers")]
    #[error("Failed to download internal buffer '{0}' for visualization")]
    DownloadError(&'static str),

    #[cfg(feature = "wgpu")]
    #[error("wgpu Error from scope")]
    WgpuErrorFromScope(#[from] wgpu::Error),

    /// Failed to create [`GpuProfiler`].
    /// See [`wgpu_profiler::CreationError`] for more information.
    #[cfg(feature = "wgpu-profiler")]
    #[error("Couldn't create wgpu profiler")]
    #[doc(hidden)] // End-users of Vello should not have `wgpu-profiler` enabled.
    ProfilerCreationError(#[from] wgpu_profiler::CreationError),

    /// Failed to compile the shaders.
    #[cfg(feature = "hot_reload")]
    #[error("Failed to compile shaders:\n{0}")]
    #[doc(hidden)] // End-users of Vello should not have `hot_reload` enabled.
    ShaderCompilation(#[from] vello_shaders::compile::ErrorVec),
}

#[cfg_attr(
    not(feature = "wgpu"),
    expect(dead_code, reason = "this can be unused when wgpu feature is not used")
)]
pub(crate) type Result<T, E = Error> = std::result::Result<T, E>;

/// Renders a scene into a texture or surface.
///
/// Currently, each renderer only supports a single surface format, if it
/// supports drawing to surfaces at all.
/// This is an assumption which is known to be limiting, and is planned to change.
#[cfg(feature = "wgpu")]
pub struct Renderer {
    options: RendererOptions,
    engine: WgpuEngine,
    resolver: Resolver,
    image_atlas: Option<recording::ImageProxy>,
    shaders: FullShaders,
    /// The least the bump-allocated buffers get: the largest sizes any frame
    /// so far was given, grown whenever a frame ran out of one (see
    /// [`Renderer::render_to_texture`]). It never shrinks.
    bump_sizes: vello_encoding::BumpSizes,
    /// The counters of frames submitted and not yet read, oldest first, at
    /// most `bump_readback_limit` of them.
    bump_readbacks: VecDeque<BumpReadback>,
    /// [`MAX_BUMP_READBACKS`], unless a test says otherwise.
    bump_readback_limit: usize,
    /// A frame was found to have overflowed, and [`Renderer::check_frames`]
    /// has not said so yet.
    frame_lost: bool,
    /// The [`BumpReport`]s kept since [`Renderer::record_bump_reports`].
    bump_reports: bump_report::Recorder,
    #[cfg(feature = "debug_layers")]
    debug: debug::DebugRenderer,
    #[cfg(feature = "wgpu-profiler")]
    #[doc(hidden)] // End-users of Vello should not have `wgpu-profiler` enabled.
    /// The profiler used with events for this renderer. This is *not* treated as public API.
    pub profiler: GpuProfiler,
    #[cfg(feature = "wgpu-profiler")]
    #[doc(hidden)] // End-users of Vello should not have `wgpu-profiler` enabled.
    /// The results from profiling. This is *not* treated as public API.
    pub profile_result: Option<Vec<wgpu_profiler::GpuTimerQueryResult>>,
}
// This is not `Send` (or `Sync`) on WebAssembly as the
// underlying wgpu types are not. This can be enabled with the
// `fragile-send-sync-non-atomic-wasm` feature in wgpu.
// See https://github.com/gfx-rs/wgpu/discussions/4127 for
// further discussion of this topic.
#[cfg(all(feature = "wgpu", not(target_arch = "wasm32")))]
static_assertions::assert_impl_all!(Renderer: Send);

/// Parameters used in a single render that are configurable by the client.
///
/// These are used in [`Renderer::render_to_texture`].
pub struct RenderParams {
    /// The background color applied to the target. This value is only applicable to the full
    /// pipeline.
    pub base_color: peniko::Color,

    /// Dimensions of the rasterization target
    pub width: u32,
    pub height: u32,

    /// The anti-aliasing algorithm. The selected algorithm must have been initialized while
    /// constructing the `Renderer`.
    pub antialiasing_method: AaConfig,
}

#[cfg(feature = "wgpu")]
/// Options which are set at renderer creation time, used in [`Renderer::new`].
pub struct RendererOptions {
    /// If true, run all stages up to fine rasterization on the CPU.
    ///
    /// This is not a recommended configuration as it is expected to have poor performance,
    /// but it can be useful for debugging.
    // TODO: Consider evolving this so that the CPU stages can be configured dynamically via
    // `RenderParams`.
    pub use_cpu: bool,

    /// Represents the enabled set of AA configurations. This will be used to determine which
    /// pipeline permutations should be compiled at startup.
    ///
    /// By default this will be all modes, to support the widest range of.
    /// It is recommended that most users configure this.
    pub antialiasing_support: AaSupport,

    /// How many threads to use for initialisation of shaders.
    ///
    /// Use `Some(1)` to use a single thread. This is recommended when on macOS
    /// (see <https://github.com/bevyengine/bevy/pull/10812#discussion_r1496138004>)
    ///
    /// Set to `None` to use a heuristic which will use many but not all threads
    ///
    /// Has no effect on WebAssembly
    ///
    /// Will default to `None` on most platforms, `Some(1)` on macOS.
    pub num_init_threads: Option<NonZeroUsize>,

    /// The pipeline cache to use when creating the shaders.
    ///
    /// For much more discussion of expected usage patterns, see the documentation on that type.
    pub pipeline_cache: Option<wgpu::PipelineCache>,
}

#[cfg(feature = "wgpu")]
impl Default for RendererOptions {
    fn default() -> Self {
        Self {
            use_cpu: false,
            antialiasing_support: AaSupport::all(),
            #[cfg(target_os = "macos")]
            num_init_threads: NonZeroUsize::new(1),
            #[cfg(not(target_os = "macos"))]
            num_init_threads: None,
            pipeline_cache: None,
        }
    }
}

/// What has become of the frames [`Renderer::render_to_texture`] submitted,
/// from [`Renderer::check_frames`].
#[cfg(feature = "wgpu")]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FrameCheck {
    /// Every frame has been accounted for, and none needs drawing again.
    Settled,
    /// Some frames' counters are still on their way back: check again later.
    Pending,
    /// A frame overflowed and was left undrawn, and its buffers have been
    /// grown, or its counters could not be had and it may have. Whatever it
    /// drew is wrong until it is drawn again. A check after this one says how
    /// the frames still in flight fared.
    Lost,
}

/// How many frames' counters can be on their way back at once. A split frame
/// counts each of its slices. A frame finding this many pushes out the oldest,
/// whose counters are then never seen, so that frame counts as lost: a later
/// frame does not redraw what it drew when that was a kept slice.
#[cfg(feature = "wgpu")]
const MAX_BUMP_READBACKS: usize = 64;

/// A frame's allocation counters on their way back from the GPU. Dropping this
/// frees the download.
#[cfg(feature = "wgpu")]
struct BumpReadback {
    download: wgpu::Buffer,
    /// Set by the map callback: whether the download mapped.
    mapped: Arc<OnceLock<bool>>,
    /// The config the frame was recorded with.
    config: ConfigUniform,
    /// Its scene's deepest nesting of clips and layers, for a [`BumpReport`].
    max_open_clips: u32,
}

/// The largest buffer a bump allocation can use on `device`.
#[cfg(feature = "wgpu")]
fn max_bump_bytes(device: &Device) -> u64 {
    let limits = device.limits();
    limits
        .max_storage_buffer_binding_size
        .min(limits.max_buffer_size)
}

/// The allocation counters in a mapped download, which this unmaps.
#[cfg(feature = "wgpu")]
fn read_bump_download(download: &wgpu::Buffer) -> Option<BumpAllocators> {
    let bump = {
        let mapped = download.slice(..).get_mapped_range().ok()?;
        bytemuck::pod_read_unaligned(&mapped[..size_of::<BumpAllocators>()])
    };
    download.unmap();
    Some(bump)
}

#[cfg(feature = "wgpu")]
struct RenderResult {
    bump: Option<BumpAllocators>,
    #[cfg(feature = "debug_layers")]
    captured: Option<render::CapturedBuffers>,
}

#[cfg(feature = "wgpu")]
impl Renderer {
    /// Creates a new renderer for the specified device.
    pub fn new(device: &Device, options: RendererOptions) -> Result<Self> {
        let mut engine = WgpuEngine::new(options.use_cpu, options.pipeline_cache.clone());
        // If we are running in parallel (i.e. the number of threads is not 1)
        if options.num_init_threads != NonZeroUsize::new(1) {
            #[cfg(not(target_arch = "wasm32"))]
            engine.use_parallel_initialisation();
        }
        let shaders = shaders::full_shaders(device, &mut engine, &options)?;
        #[cfg(not(target_arch = "wasm32"))]
        engine.build_shaders_if_needed(device, options.num_init_threads);
        #[cfg(feature = "debug_layers")]
        let debug = debug::DebugRenderer::new(device, wgpu::TextureFormat::Rgba8Unorm, &mut engine);

        Ok(Self {
            options,
            engine,
            resolver: Resolver::new(),
            image_atlas: None,
            shaders,
            bump_sizes: vello_encoding::BumpSizes::default(),
            bump_readbacks: VecDeque::with_capacity(MAX_BUMP_READBACKS),
            bump_readback_limit: MAX_BUMP_READBACKS,
            frame_lost: false,
            bump_reports: bump_report::Recorder::default(),
            #[cfg(feature = "debug_layers")]
            debug,
            #[cfg(feature = "wgpu-profiler")]
            profiler: GpuProfiler::new(device, GpuProfilerSettings::default())?,
            #[cfg(feature = "wgpu-profiler")]
            profile_result: None,
        })
    }

    /// Renders a scene to the target texture.
    ///
    /// The texture is assumed to be of the specified dimensions and have been created with
    /// the [`wgpu::TextureFormat::Rgba8Unorm`] format and the [`wgpu::TextureUsages::STORAGE_BINDING`]
    /// flag set.
    ///
    /// If you want to render Vello content to a surface (such as in a UI toolkit), you have two options:
    /// 1) Render to an intermediate texture, which is the same size as the surface.
    ///    You would then use [`TextureBlitter`][wgpu::util::TextureBlitter] to blit the rendered result from
    ///    that texture to the surface.
    ///    This pattern is supported by the [`util`] module.
    /// 2) Call `render_to_texture` directly on the [`SurfaceTexture`][wgpu::SurfaceTexture]'s texture, if
    ///    it has the right usages. This should generally be avoided, as some GPUs assume that you will not
    ///    be rendering to the surface using a compute pipeline, and optimise accordingly.
    ///
    /// This submits the frame and returns; it does not wait for the GPU.
    ///
    /// A frame whose GPU stages run out of one of their bump-allocated buffers
    /// (many clip layers, a dense scene) is not drawn. Those buffers start at
    /// sizes that follow the target and the scene
    /// ([`BumpSizes::initial`](vello_encoding::BumpSizes::initial)), calibrated
    /// so that an ordinary frame fits with a margin, and never get smaller than
    /// what an earlier frame was given. (With [`RendererOptions::use_cpu`] and
    /// on WebAssembly, where no counters come back, they keep Vello's fixed
    /// sizes, raised to floors that follow the target.) So that one that does
    /// not fit is not silent, except with [`RendererOptions::use_cpu`] and on
    /// WebAssembly each frame's allocation counters are read back; they arrive after the GPU has
    /// finished it, and this and [`Self::check_frames`] collect them without
    /// blocking. A frame that overflowed has the buffers it ran out of grown to
    /// what it asked for, and the grown sizes are kept for every later frame.
    ///
    /// The frame that overflowed has by then gone to the target undrawn, as
    /// have any others submitted before its counters came back: on a
    /// transparent target that is a blank frame, and nothing here draws it
    /// again. A caller that does not draw every frame anyway asks
    /// [`Self::check_frames`] until it stops saying
    /// [`Pending`](FrameCheck::Pending), and draws again on
    /// [`Lost`](FrameCheck::Lost).
    pub fn render_to_texture(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        texture: &TextureView,
        params: &RenderParams,
    ) -> Result<()> {
        let robust = !self.options.use_cpu && !cfg!(target_arch = "wasm32");
        let max_bytes = max_bump_bytes(device);
        self.collect_bump_readbacks(device, max_bytes);
        let mut render = Render::new();
        if robust {
            // Sized from the target and the scene; an overflow is found from
            // the counters and grown from.
            render.bump_limit = Some(max_bytes);
        } else {
            // No counters come back, so an overflow would go unseen: Vello's
            // sizes, raised to floors that follow the target.
            self.bump_sizes = self.bump_sizes.at_least(vello_encoding::BumpSizes::floors(
                params.width,
                params.height,
                max_bytes,
            ));
        }
        render.bump_sizes = Some(self.bump_sizes);
        let mut recording = render.render_encoding_coarse(
            scene.encoding(),
            &mut self.resolver,
            &self.shaders,
            &mut self.image_atlas,
            params,
            robust,
        );
        let target = render.out_image();
        let bump_buf = render.bump_buf();
        let gpu_config = render.gpu_config.expect("set by the coarse recording");
        // The sizes this frame was given are the least every later one gets:
        // a partial frame, a slice and a full frame of one window share them.
        self.bump_sizes = self
            .bump_sizes
            .at_least(vello_encoding::BumpSizes::of(&gpu_config));
        render.record_fine(&self.shaders, &mut recording);
        let external_resources = [ExternalResource::Image(target, texture)];
        let submitted = self.engine.run_recording(
            device,
            queue,
            &recording,
            &external_resources,
            "render_to_texture",
            #[cfg(feature = "wgpu-profiler")]
            &mut self.profiler,
        );
        if let Err(e) = submitted {
            // The download may be in place by the time a recording fails.
            self.engine.free_download(bump_buf);
            return Err(e);
        }
        // N.B. This is horrible; this integration of wgpu-profiler really needs some work...
        #[cfg(feature = "wgpu-profiler")]
        {
            self.profiler.end_frame().unwrap();
            if let Some(result) = self
                .profiler
                .process_finished_frame(queue.get_timestamp_period())
            {
                self.profile_result = Some(result);
            }
        }
        if robust {
            self.queue_bump_readback(bump_buf, gpu_config, scene.encoding().max_open_clips);
        }
        Ok(())
    }

    /// What has become of the frames [`Self::render_to_texture`] submitted.
    /// This never waits for the GPU. See [`FrameCheck`].
    ///
    /// [`Lost`](FrameCheck::Lost) means drawing the frame again can help: a
    /// frame that needs more than the device's largest buffer is logged and
    /// not reported, since it would fail again every time.
    ///
    /// This does not report a lost device: it polls the device, and on a lost
    /// device `Device::poll` panics rather than returning an error (wgpu holds
    /// it fatal; seen on Vulkan). A host that means to survive a device loss
    /// cannot do so past a call to this or to [`Self::render_to_texture`]
    /// while readbacks are out.
    pub fn check_frames(&mut self, device: &Device) -> FrameCheck {
        self.collect_bump_readbacks(device, max_bump_bytes(device));
        if std::mem::take(&mut self.frame_lost) {
            FrameCheck::Lost
        } else if self.bump_readbacks.is_empty() {
            FrameCheck::Settled
        } else {
            FrameCheck::Pending
        }
    }

    /// Keep a [`BumpReport`] of each frame whose counters
    /// [`Self::check_frames`] takes in from now on, at most `capacity` of
    /// them (the newest), until [`Self::take_bump_reports`] drains them. 0
    /// stops and drops those kept. Off by default; off, nothing is kept.
    ///
    /// Only frames of the robust path report: on the GPU, not `use_cpu`, not
    /// on WebAssembly. For measuring the rules that size the buffers
    /// ([`vello_encoding::BumpSizes::initial`]).
    pub fn record_bump_reports(&mut self, capacity: usize) {
        self.bump_reports.set_capacity(capacity);
    }

    /// The [`BumpReport`]s kept since the last take, oldest first. Recording
    /// goes on; this only drains.
    pub fn take_bump_reports(&mut self) -> Vec<BumpReport> {
        self.bump_reports.take()
    }

    /// Limits the frames whose counters can be on their way back, so that a
    /// test can have one pushed out without outrunning the GPU 64 times.
    #[doc(hidden)]
    pub fn set_bump_readback_limit(&mut self, limit: usize) {
        self.bump_readback_limit = limit.max(1);
    }

    /// Starts reading back the counters of a frame that has been submitted.
    /// The answer arrives when the device is next polled.
    fn queue_bump_readback(
        &mut self,
        buf: recording::BufferProxy,
        config: ConfigUniform,
        max_open_clips: u32,
    ) {
        let Some(download) = self.engine.take_download(buf) else {
            return;
        };
        while self.bump_readbacks.len() >= self.bump_readback_limit {
            self.bump_readbacks.pop_front();
            log::debug!("vello: a frame's counters were pushed out unread");
            self.frame_lost = true;
        }
        let mapped = Arc::new(OnceLock::new());
        let answer = mapped.clone();
        download
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = answer.set(result.is_ok());
            });
        self.bump_readbacks.push_back(BumpReadback {
            download,
            mapped,
            config,
            max_open_clips,
        });
    }

    /// Takes in the counters that have come back, without waiting for the ones
    /// that have not, growing the buffers for any frame that overflowed.
    fn collect_bump_readbacks(&mut self, device: &Device, max_bytes: u64) {
        if self.bump_readbacks.is_empty() {
            return;
        }
        // An error here is a poll that can be made again. On a lost device
        // there is no error: `Device::poll` panics, wgpu holding that fatal.
        let _ = device.poll(wgpu::PollType::Poll);
        while let Some(&mapped) = self.bump_readbacks.front().and_then(|r| r.mapped.get()) {
            let readback = self.bump_readbacks.pop_front().expect("just looked at");
            let bump = mapped
                .then(|| read_bump_download(&readback.download))
                .flatten();
            let Some(bump) = bump else {
                // Nothing says this frame fitted.
                log::warn!("vello: a frame's counters could not be read");
                self.frame_lost = true;
                continue;
            };
            let config = &readback.config;
            self.bump_reports
                .record(&bump, config, readback.max_open_clips, max_bytes);
            let (width, height) = (config.target_width, config.target_height);
            let had = vello_encoding::BumpSizes::of(config);
            // What the initial sizes are derived from, beside what the frame
            // used: the counts a recalibration of `BumpSizes::initial` reads.
            let layout = &config.layout;
            log::debug!(
                "vello: a {width}x{height} frame of {} path tags, {} paths, {} draw objects and {} clips used {bump:?} of {had:?}",
                layout.path_tags_size(),
                layout.n_paths,
                layout.n_draw_objects,
                layout.n_clips
            );
            if bump.failed == 0 {
                continue;
            }
            let grown = vello_encoding::BumpSizes::grown_for(config, &bump, max_bytes)
                .at_least(self.bump_sizes);
            if grown == had {
                log::warn!(
                    "vello: a {width}x{height} frame needs more than the device's largest buffer ({max_bytes} bytes); it was not drawn ({bump:?})"
                );
                continue;
            }
            log::info!(
                "vello: a {width}x{height} frame ran out of buffer space and went out undrawn ({bump:?}); the buffers are now {grown:?}"
            );
            self.bump_sizes = grown;
            self.frame_lost = true;
        }
    }

    /// Overwrite `image` with `texture`.
    ///
    /// Most users should prefer [`register_texture`](Self::register_texture), which
    /// ergonomically encapulates these requirements.
    ///
    /// `texture` must have the [`wgpu::TextureFormat::Rgba8Unorm`] format and
    /// the [`wgpu::TextureUsages::COPY_SRC`] flag set. The `Rgba8UnormSrgb` format
    /// might also be supported, but this is not tested.
    ///
    /// The given `Texture`'s data will be copied into the slot in Vello's image
    /// atlas where the image would be placed each frame.
    /// This has the effect that wherever you request the image to be drawn (e.g.
    /// through [`Scene::draw_image`]), the data from the texture will be used instead.
    ///
    /// Correct behaviour is not guaranteed if the texture does not have the same
    /// dimensions as the image, nor if an image which uses the same [blob] but different
    /// dimensions would be rendered.
    ///
    /// [blob]: peniko::ImageData::data
    pub fn override_image(
        &mut self,
        image: &ImageData,
        texture: Option<wgpu::TexelCopyTextureInfoBase<wgpu::Texture>>,
    ) -> Option<wgpu::TexelCopyTextureInfoBase<wgpu::Texture>> {
        self.resolver.mark_image_dirty(image);
        match texture {
            Some(texture) => self.engine.image_overrides.insert(image.data.id(), texture),
            None => self.engine.image_overrides.remove(&image.data.id()),
        }
    }

    /// Marks `image` as dirty in the atlas cache, so its override texture will be recopied on
    /// the next render that uses it.
    ///
    /// Call this for any `image` whose existing override texture has changed since the last render.
    /// Otherwise, stale image data from the atlas may be used.
    pub fn mark_override_image_dirty(&mut self, image: &ImageData) {
        self.resolver.mark_image_dirty(image);
    }

    /// Register a [`wgpu::Texture`] with Vello, to allow drawing GPU-resident data.
    ///
    /// The returned `Image` can be used in [`Scene`]s (only those rendered with this `Renderer`)
    /// Rendering Scenes which use this `Image` with other `Renderer`s will panic.
    ///  
    /// `texture` must have the [`wgpu::TextureFormat::Rgba8Unorm`] format and
    /// the [`wgpu::TextureUsages::COPY_SRC`] flag set. This is because the data will
    /// be copied into Vello's image atlas at the start of each frame.
    /// The `Rgba8UnormSrgb` format might also be supported, but this is not tested.
    /// The texture is assumed to have unpremultiplied alpha.
    ///
    /// This is a utility wrapper around [`override_image`](Self::override_image).
    /// For greater control, use that method.
    ///
    /// If the texture is no longer active then it should be unregistered using [`unregister_texture`](Self::unregister_texture)
    pub fn register_texture(&mut self, texture: wgpu::Texture) -> ImageData {
        // Create a fake, empty blob which will be used to back the returned image
        // This image data will never be read by Vello, due to being added to
        // image_overrides, below.
        let fake_blob = peniko::Blob::new(Arc::new(&[]));

        let image = ImageData {
            data: fake_blob,
            format: peniko::ImageFormat::Rgba8,
            alpha_type: peniko::ImageAlphaType::Alpha,
            width: texture.width(),
            height: texture.height(),
        };

        // For this utility API, we take the full texture and use the base layer and mip level
        let texture_base = wgpu::TexelCopyTextureInfoBase {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        };

        // We overwrite any attempt to use the fake blob, instead reading from the texture.
        self.override_image(&image, Some(texture_base));

        image
    }

    /// Unregister a [`wgpu::Texture`] that was registered with [`register_texture`](Self::register_texture).
    pub fn unregister_texture(&mut self, handle: ImageData) {
        self.override_image(&handle, None);
    }

    /// Reload the shaders. This should only be used during `vello` development
    #[cfg(feature = "hot_reload")]
    #[doc(hidden)] // End-users of Vello should not have `hot_reload` enabled.
    pub async fn reload_shaders(&mut self, device: &Device) -> Result<(), Error> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut engine = WgpuEngine::new(self.options.use_cpu, self.options.pipeline_cache.clone());
        // We choose not to initialise these shaders in parallel, to ensure the error scope works correctly
        let shaders = shaders::full_shaders(device, &mut engine, &self.options)?;
        #[cfg(feature = "debug_layers")]
        let debug = debug::DebugRenderer::new(device, wgpu::TextureFormat::Rgba8Unorm, &mut engine);
        let error = scope.pop().await;
        if let Some(error) = error {
            return Err(error.into());
        }
        self.engine = engine;
        self.image_atlas = None;
        self.shaders = shaders;
        #[cfg(feature = "debug_layers")]
        {
            self.debug = debug;
        }
        Ok(())
    }

    /// Renders a scene to the target texture using an async pipeline.
    ///
    /// Almost all consumers should prefer [`Self::render_to_texture`].
    ///
    /// The return value is the value of the `BumpAllocators` in this rendering, which is currently used
    /// for debug output.
    ///
    /// This return type is not stable, and will likely be changed when a more principled way to access
    /// relevant statistics is implemented
    #[cfg_attr(docsrs, doc(hidden))]
    #[deprecated(
        note = "render_to_texture should be preferred, as the _async version has no stability guarantees"
    )]
    pub async fn render_to_texture_async(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        texture: &TextureView,
        params: &RenderParams,
        debug_layers: DebugLayers,
    ) -> Result<Option<BumpAllocators>> {
        if cfg!(not(feature = "debug_layers")) && !debug_layers.is_empty() {
            static HAS_WARNED: AtomicBool = AtomicBool::new(false);
            if !HAS_WARNED.swap(true, std::sync::atomic::Ordering::Release) {
                log::warn!(
                    "Requested debug layers {debug_layers:?} but `debug_layers` feature is not enabled"
                );
            }
        }

        let result = self
            .render_to_texture_async_internal(device, queue, scene, texture, params)
            .await?;

        #[cfg(feature = "debug_layers")]
        {
            let mut recording = Recording::default();
            let target_proxy = recording::ImageProxy::new(
                params.width,
                params.height,
                recording::ImageFormat::Rgba8,
            );
            if let Some(captured) = result.captured {
                let bump = result.bump.as_ref().unwrap();
                // TODO: We could avoid this download if `DebugLayers::VALIDATION` is unset.
                let downloads = DebugDownloads::map(&self.engine, &captured, bump).await?;
                self.debug.render(
                    &mut recording,
                    target_proxy,
                    &captured,
                    bump,
                    params,
                    &downloads,
                    debug_layers,
                );

                // TODO: this sucks. better to release everything in a helper
                // TODO: it would be much better to have a way to safely destroy a buffer.
                self.engine.free_download(captured.lines);
                captured.release_buffers(&mut recording);
            }
            let external_resources = [ExternalResource::Image(target_proxy, texture)];
            self.engine.run_recording(
                device,
                queue,
                &recording,
                &external_resources,
                "render_to_texture_async debug layers",
                #[cfg(feature = "wgpu-profiler")]
                &mut self.profiler,
            )?;
        }

        #[cfg(feature = "wgpu-profiler")]
        {
            self.profiler.end_frame().unwrap();
            if let Some(result) = self
                .profiler
                .process_finished_frame(queue.get_timestamp_period())
            {
                self.profile_result = Some(result);
            }
        }

        Ok(result.bump)
    }

    async fn render_to_texture_async_internal(
        &mut self,
        device: &Device,
        queue: &Queue,
        scene: &Scene,
        texture: &TextureView,
        params: &RenderParams,
    ) -> Result<RenderResult> {
        let mut render = Render::new();
        let encoding = scene.encoding();
        // TODO: turn this on; the download feature interacts with CPU dispatch.
        // Currently this is always enabled when the `debug_layers` setting is enabled as the bump
        // counts are used for debug visualiation.
        let robust = cfg!(feature = "debug_layers");
        let recording = render.render_encoding_coarse(
            encoding,
            &mut self.resolver,
            &self.shaders,
            &mut self.image_atlas,
            params,
            robust,
        );
        let target = render.out_image();
        let bump_buf = render.bump_buf();
        #[cfg(feature = "debug_layers")]
        let captured = render.take_captured_buffers();
        self.engine.run_recording(
            device,
            queue,
            &recording,
            &[],
            "t_async_coarse",
            #[cfg(feature = "wgpu-profiler")]
            &mut self.profiler,
        )?;

        let mut bump: Option<BumpAllocators> = None;
        if let Some(bump_buf) = self.engine.get_download(bump_buf) {
            let buf_slice = bump_buf.slice(..);
            let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
            buf_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
            receiver.receive().await.expect("channel was closed")?;
            let mapped = buf_slice.get_mapped_range().unwrap();
            bump = Some(bytemuck::pod_read_unaligned(&mapped));
        }
        // TODO: apply logic to determine whether we need to rerun coarse, and also
        // allocate the blend stack as needed.
        self.engine.free_download(bump_buf);
        // Maybe clear to reuse allocation?
        let mut recording = Recording::default();
        render.record_fine(&self.shaders, &mut recording);
        let external_resources = [ExternalResource::Image(target, texture)];
        self.engine.run_recording(
            device,
            queue,
            &recording,
            &external_resources,
            "t_async_fine",
            #[cfg(feature = "wgpu-profiler")]
            &mut self.profiler,
        )?;
        Ok(RenderResult {
            bump,
            #[cfg(feature = "debug_layers")]
            captured,
        })
    }
}
#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
pub(crate) struct DebugDownloads<'a> {
    pub lines: wgpu::BufferSlice<'a>,
}

#[cfg(all(feature = "debug_layers", feature = "wgpu"))]
impl<'a> DebugDownloads<'a> {
    pub async fn map(
        engine: &'a WgpuEngine,
        captured: &render::CapturedBuffers,
        bump: &BumpAllocators,
    ) -> Result<DebugDownloads<'a>> {
        use vello_encoding::LineSoup;

        let Some(lines_buf) = engine.get_download(captured.lines) else {
            return Err(Error::DownloadError("linesoup"));
        };

        let lines = lines_buf.slice(..bump.lines as u64 * size_of::<LineSoup>() as u64);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        lines.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        receiver.receive().await.expect("channel was closed")?;
        Ok(Self { lines })
    }
}

/// The bind-group cache (rinch #2469). Needs a GPU adapter:
/// `cargo test -p vello -- --ignored bind_group`.
#[cfg(all(test, feature = "wgpu", not(target_arch = "wasm32")))]
mod bind_group_cache_tests {
    use super::*;
    use peniko::kurbo::{Affine, Rect, RoundedRect};
    use peniko::{Color, Fill};

    fn gpu() -> (Device, Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("these tests need a GPU adapter");
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            .expect("device")
    }

    fn scene(dx: f64) -> Scene {
        let mut s = Scene::new();
        s.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::from_rgba8(40, 40, 50, 255),
            None,
            &Rect::new(0.0, 0.0, 256.0, 64.0),
        );
        s.fill(
            Fill::NonZero,
            Affine::IDENTITY,
            Color::from_rgba8(200, 200, 210, 255),
            None,
            &RoundedRect::new(8.0 + dx, 8.0, 120.0 + dx, 40.0, 6.0),
        );
        s
    }

    #[test]
    #[ignore = "needs a GPU adapter"]
    fn a_repeated_frame_makes_only_the_bind_groups_of_its_new_resources() {
        let (device, queue) = gpu();
        let mut renderer = Renderer::new(
            &device,
            RendererOptions {
                use_cpu: false,
                antialiasing_support: AaSupport::area_only(),
                num_init_threads: NonZeroUsize::new(1),
                pipeline_cache: None,
            },
        )
        .unwrap();
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 256,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let params = RenderParams {
            base_color: Color::BLACK,
            width: 256,
            height: 64,
            antialiasing_method: AaConfig::Area,
        };
        let render = |r: &mut Renderer, dx: f64| {
            r.render_to_texture(&device, &queue, &scene(dx), &view, &params)
                .unwrap();
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            assert_eq!(r.check_frames(&device), FrameCheck::Settled);
            r.engine.bind_groups.created
        };
        // Two frames to settle the pool's buffers, then a different scene of the same shape.
        render(&mut renderer, 0.0);
        let settled = render(&mut renderer, 1.0);
        let next = render(&mut renderer, 2.0);
        let dispatches = renderer.engine.bind_groups.groups.len() as u64;
        // Only the dispatches that bind a texture made for the frame (the gradient ramps, which
        // Vello uploads each frame) need a new group; every other one is reused.
        assert!(
            next - settled <= 2,
            "a repeated frame made {} bind groups ({} cached)",
            next - settled,
            dispatches
        );
        assert!(dispatches >= 8, "only {dispatches} bind groups cached");

        // Groups not used for KEEP_RECORDINGS recordings are dropped.
        for _ in 0..wgpu_engine::KEEP_BIND_GROUPS_FOR_RECORDINGS {
            renderer.engine.bind_groups.end_recording();
        }
        assert_eq!(renderer.engine.bind_groups.groups.len(), 0);
    }

    fn renderer(device: &Device) -> Renderer {
        Renderer::new(
            device,
            RendererOptions {
                use_cpu: false,
                antialiasing_support: AaSupport::area_only(),
                num_init_threads: NonZeroUsize::new(1),
                pipeline_cache: None,
            },
        )
        .unwrap()
    }

    /// A target texture made per frame, as a window being resized makes them, is released once
    /// the caller drops it: no cached group holds its view (the review of joeleaver/vello-fork#10
    /// measured 16 MB more GPU memory per 1440p resize while one did).
    #[test]
    #[ignore = "needs a GPU adapter with an allocator report (Vulkan, DX12, Metal)"]
    fn a_target_made_per_frame_is_released() {
        let (device, queue) = gpu();
        let mut renderer = renderer(&device);
        let (w, h) = (1024, 1024);
        let params = RenderParams {
            base_color: Color::BLACK,
            width: w,
            height: h,
            antialiasing_method: AaConfig::Area,
        };
        let frame = |r: &mut Renderer, dx: f64| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("per-frame target"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            r.render_to_texture(&device, &queue, &scene(dx), &view, &params)
                .unwrap();
            drop((view, texture));
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let _ = r.check_frames(&device);
            device
                .generate_allocator_report()
                .expect("an allocator report")
                .total_allocated_bytes
        };
        frame(&mut renderer, 0.0);
        let settled = frame(&mut renderer, 1.0);
        let mut last = settled;
        for i in 0..12 {
            last = frame(&mut renderer, 2.0 + f64::from(i));
        }
        // Each target is 4 MiB: twelve kept alive would be 48 MiB more.
        assert!(
            last <= settled + (2 << 20),
            "GPU memory grew from {settled} to {last} bytes over 12 per-frame targets"
        );
    }
}
