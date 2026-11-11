// Copyright 2023 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

use crate::SegmentCount;

use super::{
    BinHeader, Clip, ClipBbox, ClipBic, ClipElement, DrawBbox, DrawMonoid, Layout, LineSoup, Path,
    PathBbox, PathMonoid, PathSegment, Tile,
};
use bytemuck::{Pod, Zeroable};

mod initial;

const TILE_WIDTH: u32 = 16;
const TILE_HEIGHT: u32 = 16;

// TODO: Obtain these from the vello_shaders crate
pub(crate) const PATH_REDUCE_WG: u32 = 256;
const PATH_BBOX_WG: u32 = 256;
const FLATTEN_WG: u32 = 256;
const CLIP_REDUCE_WG: u32 = 256;

/// Counters for tracking dynamic allocation on the GPU.
///
/// This must be kept in sync with the struct in `shader/shared/bump.wgsl`
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct BumpAllocators {
    pub failed: u32,
    // Final needed dynamic size of the buffers. If any of these are larger
    // than the corresponding `_size` element reallocation needs to occur.
    pub binning: u32,
    pub ptcl: u32,
    pub tile: u32,
    pub seg_counts: u32,
    pub segments: u32,
    pub blend: u32,
    pub lines: u32,
}

#[derive(Default)]
pub struct BumpAllocatorMemory {
    pub total: u32,
    pub binning: BufferSize<u32>,
    pub ptcl: BufferSize<u32>,
    pub tile: BufferSize<Tile>,
    pub seg_counts: BufferSize<SegmentCount>,
    pub segments: BufferSize<PathSegment>,
    pub lines: BufferSize<LineSoup>,
}

impl BumpAllocators {
    pub fn memory(&self) -> BumpAllocatorMemory {
        let binning = BufferSize::new(self.binning);
        let ptcl = BufferSize::new(self.ptcl);
        let tile = BufferSize::new(self.tile);
        let seg_counts = BufferSize::new(self.seg_counts);
        let segments = BufferSize::new(self.segments);
        let lines = BufferSize::new(self.lines);
        BumpAllocatorMemory {
            total: binning.size_in_bytes()
                + ptcl.size_in_bytes()
                + tile.size_in_bytes()
                + seg_counts.size_in_bytes()
                + segments.size_in_bytes()
                + lines.size_in_bytes(),
            binning,
            ptcl,
            tile,
            seg_counts,
            segments,
            lines,
        }
    }
}

impl std::fmt::Display for BumpAllocatorMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "\n \
                 \tTotal:\t\t\t{} bytes ({:.2} KB | {:.2} MB)\n\
                 \tBinning\t\t\t{} elements ({} bytes)\n\
                 \tPTCL\t\t\t{} elements ({} bytes)\n\
                 \tTile:\t\t\t{} elements ({} bytes)\n\
                 \tSegment Counts:\t\t{} elements ({} bytes)\n\
                 \tSegments:\t\t{} elements ({} bytes)\n\
                 \tLines:\t\t\t{} elements ({} bytes)",
            self.total,
            self.total as f32 / (1 << 10) as f32,
            self.total as f32 / (1 << 20) as f32,
            self.binning.len(),
            self.binning.size_in_bytes(),
            self.ptcl.len(),
            self.ptcl.size_in_bytes(),
            self.tile.len(),
            self.tile.size_in_bytes(),
            self.seg_counts.len(),
            self.seg_counts.size_in_bytes(),
            self.segments.len(),
            self.segments.size_in_bytes(),
            self.lines.len(),
            self.lines.size_in_bytes()
        )
    }
}

/// Storage of indirect dispatch size values.
///
/// The original plan was to reuse [`BumpAllocators`], but the WebGPU compatible
/// usage list rules forbid that being used as indirect counts while also
/// bound as writable.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct IndirectCount {
    pub count_x: u32,
    pub count_y: u32,
    pub count_z: u32,
    pub pad0: u32,
}

/// Uniform render configuration data used by all GPU stages.
///
/// This data structure must be kept in sync with the definition in
/// `shaders/shared/config.wgsl`.
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
#[repr(C)]
pub struct ConfigUniform {
    /// Width of the scene in tiles.
    pub width_in_tiles: u32,
    /// Height of the scene in tiles.
    pub height_in_tiles: u32,
    /// Width of the target in pixels.
    pub target_width: u32,
    /// Height of the target in pixels.
    pub target_height: u32,
    /// The base background color applied to the target before any blends.
    pub base_color: u32,
    /// Layout of packed scene data.
    pub layout: Layout,
    /// Size of line soup buffer allocation (in [`LineSoup`]s)
    pub lines_size: u32,
    /// Size of binning buffer allocation (in `u32`s).
    pub binning_size: u32,
    /// Size of tile buffer allocation (in [`Tile`]s).
    pub tiles_size: u32,
    /// Size of segment count buffer allocation (in [`SegmentCount`]s).
    pub seg_counts_size: u32,
    /// Size of segment buffer allocation (in [`PathSegment`]s).
    pub segments_size: u32,
    /// Size of blend spill buffer (in `u32` pixels).
    // TODO: Maybe store in TILE_WIDTH * TILE_HEIGHT blocks of pixels instead?
    pub blend_size: u32,
    /// Size of per-tile command list buffer allocation (in `u32`s).
    pub ptcl_size: u32,
}

/// CPU side setup and configuration.
#[derive(Default)]
pub struct RenderConfig {
    /// GPU side configuration.
    pub gpu: ConfigUniform,
    /// Workgroup counts for all compute pipelines.
    pub workgroup_counts: WorkgroupCounts,
    /// Sizes of all buffer resources.
    pub buffer_sizes: BufferSizes,
}

impl RenderConfig {
    pub fn new(layout: &Layout, width: u32, height: u32, base_color: &peniko::Color) -> Self {
        let new_width = width.next_multiple_of(TILE_WIDTH);
        let new_height = height.next_multiple_of(TILE_HEIGHT);
        let width_in_tiles = new_width / TILE_WIDTH;
        let height_in_tiles = new_height / TILE_HEIGHT;
        let n_path_tags = layout.path_tags_size();
        let workgroup_counts =
            WorkgroupCounts::new(layout, width_in_tiles, height_in_tiles, n_path_tags);
        let buffer_sizes = BufferSizes::new(layout, &workgroup_counts);
        Self {
            gpu: ConfigUniform {
                width_in_tiles,
                height_in_tiles,
                target_width: width,
                target_height: height,
                base_color: base_color.premultiply().to_rgba8().to_u32(),
                lines_size: buffer_sizes.lines.len(),
                binning_size: buffer_sizes.bin_data.len() - layout.bin_data_start,
                tiles_size: buffer_sizes.tiles.len(),
                seg_counts_size: buffer_sizes.seg_counts.len(),
                segments_size: buffer_sizes.segments.len(),
                blend_size: buffer_sizes.blend_spill.len(),
                ptcl_size: buffer_sizes.ptcl.len(),
                layout: *layout,
            },
            workgroup_counts,
            buffer_sizes,
        }
    }
}

impl RenderConfig {
    /// This config with its bump-allocated buffers at least as large as `min`.
    ///
    /// Vello's own sizes for these buffers are fixed guesses that do not grow
    /// with the target or the scene. A renderer that reads no counters back
    /// (`use_cpu`, WebAssembly), and so never sees an overflow, keeps them and
    /// passes here the floors its target needs ([`BumpSizes::floors`]). One that
    /// does read them back sizes the buffers exactly instead
    /// ([`Self::with_exact_bump_sizes`]).
    #[must_use]
    pub fn with_bump_sizes(mut self, min: &BumpSizes) -> Self {
        let sizes = &mut self.buffer_sizes;
        let bin_data_start = self.gpu.layout.bin_data_start;
        sizes.lines = BufferSize::new(sizes.lines.len().max(min.lines));
        sizes.bin_data = BufferSize::new(
            sizes
                .bin_data
                .len()
                .max(bin_data_start.saturating_add(min.binning)),
        );
        sizes.tiles = BufferSize::new(sizes.tiles.len().max(min.tiles));
        sizes.seg_counts = BufferSize::new(sizes.seg_counts.len().max(min.seg_counts));
        sizes.segments = BufferSize::new(sizes.segments.len().max(min.segments));
        sizes.blend_spill = BufferSize::new(sizes.blend_spill.len().max(min.blend));
        sizes.ptcl = BufferSize::new(sizes.ptcl.len().max(min.ptcl));
        self.sync_bump_sizes()
    }
}

impl RenderConfig {
    /// This config with its bump-allocated buffers at exactly `sizes`, Vello's
    /// own sizes for them set aside.
    ///
    /// For a renderer that reads every frame's counters back and grows what a
    /// frame ran out of: it starts from [`BumpSizes::initial`], which follows
    /// the target and the scene, and keeps the largest sizes any frame has had
    /// or needed. Without that readback (`use_cpu`, WebAssembly) an overflow
    /// would go unseen, and [`Self::with_bump_sizes`] keeps Vello's sizes.
    #[must_use]
    pub fn with_exact_bump_sizes(mut self, sizes: &BumpSizes) -> Self {
        let buffers = &mut self.buffer_sizes;
        let bin_data_start = self.gpu.layout.bin_data_start;
        buffers.lines = BufferSize::new(sizes.lines);
        buffers.bin_data = BufferSize::new(bin_data_start.saturating_add(sizes.binning));
        buffers.tiles = BufferSize::new(sizes.tiles);
        buffers.seg_counts = BufferSize::new(sizes.seg_counts);
        buffers.segments = BufferSize::new(sizes.segments);
        buffers.blend_spill = BufferSize::new(sizes.blend);
        buffers.ptcl = BufferSize::new(sizes.ptcl);
        self.sync_bump_sizes()
    }

    /// The GPU config's bump sizes from the buffer sizes.
    fn sync_bump_sizes(mut self) -> Self {
        let sizes = &self.buffer_sizes;
        let gpu = &mut self.gpu;
        gpu.lines_size = sizes.lines.len();
        gpu.binning_size = sizes.bin_data.len() - gpu.layout.bin_data_start;
        gpu.tiles_size = sizes.tiles.len();
        gpu.seg_counts_size = sizes.seg_counts.len();
        gpu.segments_size = sizes.segments.len();
        gpu.blend_size = sizes.blend_spill.len();
        gpu.ptcl_size = sizes.ptcl.len();
        self
    }
}

/// The lengths, in elements, of the buffers Vello's GPU stages allocate from
/// with atomic bumps, which overflow when a frame needs more than they hold.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BumpSizes {
    pub lines: u32,
    /// Of `bin_data`, past the info it starts with.
    pub binning: u32,
    pub tiles: u32,
    pub seg_counts: u32,
    pub segments: u32,
    pub blend: u32,
    /// The whole per-tile command list, its static part included.
    pub ptcl: u32,
}

/// The largest buffer size at or under `max` that the engine's pool leaves as
/// it is. The pool rounds every size up to a power of two or one and a half
/// times one (`SIZE_CLASS_BITS` in `wgpu_engine.rs`), so a buffer sized to a
/// limit that is neither is created past the limit, and `create_buffer` panics.
const fn pool_size_class_below(max: u64) -> u64 {
    if max < 2 {
        return max;
    }
    let power = 1 << (63 - max.leading_zeros());
    let between = power + power / 2;
    if max >= between { between } else { power }
}

/// The floor of the tile buffer, in tiles per tile of the target.
///
/// Vello's fixed 2^21 tiles give a 4K target 64 per target tile and an 8K one
/// 16. A 4K frame of a busy app used 9 per target tile and a 1080p one 14
/// (Pimble's shape, after rinch stopped pushing clips that cut nothing), so 32
/// is twice the worst of those. It changes nothing up to 5K.
const TILES_PER_TARGET_TILE: u32 = 32;

/// The floor of the per-tile command list, in words per tile of the target,
/// the 64 the shaders reserve included.
///
/// Vello's fixed 2^23 words give a 1080p target 1035 per target tile, a 4K one
/// 259 and an 8K one 65, which leaves an 8K frame almost nothing past the
/// reserved 64. Text and boxes use 65 to 72 at 1080p and 4K, but 8 sequential
/// full-window opacity layers, a plausible page, overflow 256 at 4K, and a new
/// renderer would pay that again each time. It changes nothing up to 1440p.
const PTCL_WORDS_PER_TARGET_TILE: u32 = 512;

/// Segments a grown buffer of lines is followed by, per line, when flattening
/// ran out and the stages after it said nothing (`BumpSizes::grown_for`). In
/// rinch #2466's calibration, frames of more than a few thousand lines used at
/// most 4.6 a line (rinch's component showcase at 4K); smaller ones used more,
/// which the initial sizes' floors already hold.
const GROWN_SEGMENTS_PER_LINE: u32 = 8;

/// What the shaders' `PTCL_INITIAL_ALLOC` reserves per tile before the
/// dynamic part of the command list (`shader/shared/ptcl.wgsl`).
const PTCL_INITIAL_ALLOC: u32 = 64;

/// The stage bits of [`BumpAllocators::failed`] (`shader/shared/bump.wgsl`).
const STAGE_BINNING: u32 = 0x1;
const STAGE_TILE_ALLOC: u32 = 0x2;
const STAGE_FLATTEN: u32 = 0x4;
const STAGE_PATH_COUNT: u32 = 0x8;
const STAGE_COARSE: u32 = 0x10;

impl BumpSizes {
    /// The sizes a frame was recorded with, from its config.
    pub fn of(g: &ConfigUniform) -> Self {
        Self {
            lines: g.lines_size,
            binning: g.binning_size,
            tiles: g.tiles_size,
            seg_counts: g.seg_counts_size,
            segments: g.segments_size,
            blend: g.blend_size,
            ptcl: g.ptcl_size,
        }
    }

    /// The least a frame of a `width`x`height` target should have, whatever
    /// any frame has needed so far: what a tile buffer and a command list need
    /// grows with the target, and Vello's fixed sizes do not. Only those two
    /// buffers have floors; every other field is 0, which leaves Vello's size.
    ///
    /// Neither floor passes `max_bytes`, the device's largest storage binding,
    /// nor does the engine's pool take it past that by rounding it up.
    pub fn floors(width: u32, height: u32, max_bytes: u64) -> Self {
        let target_tiles = width
            .div_ceil(TILE_WIDTH)
            .saturating_mul(height.div_ceil(TILE_HEIGHT));
        let max_bytes = pool_size_class_below(max_bytes.min(u32::MAX as u64));
        let floor = |per_target_tile: u32, elem_bytes: usize| {
            target_tiles
                .saturating_mul(per_target_tile)
                .min((max_bytes / elem_bytes as u64) as u32)
        };
        Self {
            tiles: floor(TILES_PER_TARGET_TILE, size_of::<Tile>()),
            ptcl: floor(PTCL_WORDS_PER_TARGET_TILE, size_of::<u32>()),
            ..Self::default()
        }
    }

    /// Every field the larger of the two.
    #[must_use]
    pub fn at_least(self, other: Self) -> Self {
        Self {
            lines: self.lines.max(other.lines),
            binning: self.binning.max(other.binning),
            tiles: self.tiles.max(other.tiles),
            seg_counts: self.seg_counts.max(other.seg_counts),
            segments: self.segments.max(other.segments),
            blend: self.blend.max(other.blend),
            ptcl: self.ptcl.max(other.ptcl),
        }
    }

    /// Sizes for another try at a frame recorded with config `g` whose
    /// counters came back as `bump`: every buffer the frame overflowed made
    /// large enough for what it asked of it, with a quarter more to spare.
    /// A stage that failed without saying how much it wanted (a later stage
    /// gives up once an earlier one has failed, so its counter stops short)
    /// is doubled. No buffer grows past `max_bytes`, the device's largest
    /// storage binding; when nothing can grow, the answer equals the sizes
    /// the frame had, and drawing it again would only fail again.
    pub fn grown_for(g: &ConfigUniform, bump: &BumpAllocators, max_bytes: u64) -> Self {
        let had = Self::of(g);
        if bump.failed == 0 {
            return had;
        }
        let ptcl_static = g
            .width_in_tiles
            .saturating_mul(g.height_in_tiles)
            .saturating_mul(PTCL_INITIAL_ALLOC);
        // `BufferSize` counts bytes in a `u32`.
        let max_bytes = pool_size_class_below(max_bytes.min(u32::MAX as u64));
        let cap = |elem_bytes: usize| (max_bytes / elem_bytes as u64) as u32;
        let grow = |have: u32, wanted: u32, failed: bool, elem_bytes: usize| -> u32 {
            let target = if wanted > have {
                wanted.saturating_add(wanted / 4)
            } else if failed {
                have.saturating_mul(2)
            } else {
                have
            };
            target.min(cap(elem_bytes)).max(have)
        };
        // `binning` is the part of `bin_data` past its info, which the limit
        // on the buffer also has to hold.
        let bin_data_start = g.layout.bin_data_start;
        let cap_binning = cap(size_of::<u32>())
            .saturating_sub(bin_data_start)
            .max(had.binning);
        let f = bump.failed;
        let coarse = f & STAGE_COARSE != 0;
        let lines = grow(
            had.lines,
            bump.lines,
            f & STAGE_FLATTEN != 0,
            size_of::<LineSoup>(),
        );
        // The stages after a failed one stop short of saying what they need,
        // so a frame that ran out of lines would go on to run out of segments
        // on its next try. Those follow the lines it asked for instead.
        let after_flatten = if f & STAGE_FLATTEN != 0 {
            lines.saturating_mul(GROWN_SEGMENTS_PER_LINE)
        } else {
            0
        };
        let seg_counts = grow(
            had.seg_counts,
            bump.seg_counts.max(after_flatten),
            f & STAGE_PATH_COUNT != 0,
            size_of::<SegmentCount>(),
        );
        Self {
            lines,
            binning: grow(
                had.binning,
                bump.binning,
                f & STAGE_BINNING != 0,
                size_of::<u32>(),
            )
            .min(cap_binning),
            tiles: grow(
                had.tiles,
                bump.tile,
                f & STAGE_TILE_ALLOC != 0,
                size_of::<Tile>(),
            ),
            seg_counts,
            // Segments are allocated one per counted segment.
            segments: grow(
                had.segments,
                bump.segments.max(seg_counts),
                false,
                size_of::<PathSegment>(),
            ),
            blend: grow(
                had.blend,
                bump.blend,
                coarse && bump.blend > had.blend,
                size_of::<u32>(),
            ),
            ptcl: grow(
                had.ptcl,
                ptcl_static.saturating_add(bump.ptcl),
                coarse,
                size_of::<u32>(),
            ),
        }
    }
}

/// Type alias for a workgroup size.
pub type WorkgroupSize = (u32, u32, u32);

/// Computed sizes for all dispatches.
#[derive(Copy, Clone, Debug, Default)]
pub struct WorkgroupCounts {
    pub use_large_path_scan: bool,
    pub path_reduce: WorkgroupSize,
    pub path_reduce2: WorkgroupSize,
    pub path_scan1: WorkgroupSize,
    pub path_scan: WorkgroupSize,
    pub bbox_clear: WorkgroupSize,
    pub flatten: WorkgroupSize,
    pub draw_reduce: WorkgroupSize,
    pub draw_leaf: WorkgroupSize,
    pub clip_reduce: WorkgroupSize,
    pub clip_leaf: WorkgroupSize,
    pub binning: WorkgroupSize,
    pub tile_alloc: WorkgroupSize,
    pub path_count_setup: WorkgroupSize,
    // Note: `path_count` must use an indirect dispatch
    pub backdrop: WorkgroupSize,
    pub coarse: WorkgroupSize,
    pub path_tiling_setup: WorkgroupSize,
    // Note: `path_tiling` must use an indirect dispatch
    pub fine: WorkgroupSize,
}

impl WorkgroupCounts {
    pub fn new(
        layout: &Layout,
        width_in_tiles: u32,
        height_in_tiles: u32,
        n_path_tags: u32,
    ) -> Self {
        let n_paths = layout.n_paths;
        let n_draw_objects = layout.n_draw_objects;
        let n_clips = layout.n_clips;
        let path_tag_padded = align_up(n_path_tags, 4 * PATH_REDUCE_WG);
        let path_tag_wgs = path_tag_padded / (4 * PATH_REDUCE_WG);
        let use_large_path_scan = path_tag_wgs > PATH_REDUCE_WG;
        let reduced_size = if use_large_path_scan {
            align_up(path_tag_wgs, PATH_REDUCE_WG)
        } else {
            path_tag_wgs
        };
        let draw_object_wgs = n_draw_objects.div_ceil(PATH_BBOX_WG);
        let draw_monoid_wgs = draw_object_wgs.min(PATH_BBOX_WG);
        let flatten_wgs = n_path_tags.div_ceil(FLATTEN_WG);
        let clip_reduce_wgs = n_clips.saturating_sub(1) / CLIP_REDUCE_WG;
        let clip_wgs = n_clips.div_ceil(CLIP_REDUCE_WG);
        let path_wgs = n_paths.div_ceil(PATH_BBOX_WG);
        let width_in_bins = width_in_tiles.div_ceil(16);
        let height_in_bins = height_in_tiles.div_ceil(16);
        Self {
            use_large_path_scan,
            path_reduce: (path_tag_wgs, 1, 1),
            path_reduce2: (PATH_REDUCE_WG, 1, 1),
            path_scan1: (reduced_size / PATH_REDUCE_WG, 1, 1),
            path_scan: (path_tag_wgs, 1, 1),
            bbox_clear: (draw_object_wgs, 1, 1),
            flatten: (flatten_wgs, 1, 1),
            draw_reduce: (draw_monoid_wgs, 1, 1),
            draw_leaf: (draw_monoid_wgs, 1, 1),
            clip_reduce: (clip_reduce_wgs, 1, 1),
            clip_leaf: (clip_wgs, 1, 1),
            binning: (draw_object_wgs, 1, 1),
            tile_alloc: (path_wgs, 1, 1),
            path_count_setup: (1, 1, 1),
            backdrop: (path_wgs, 1, 1),
            coarse: (width_in_bins, height_in_bins, 1),
            path_tiling_setup: (1, 1, 1),
            fine: (width_in_tiles, height_in_tiles, 1),
        }
    }
}

/// Typed buffer size primitive.
#[derive(Copy, Clone, Eq, Default, Debug)]
pub struct BufferSize<T: Sized> {
    len: u32,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Sized> BufferSize<T> {
    /// Creates a new buffer size from number of elements.
    pub const fn new(len: u32) -> Self {
        Self {
            // Each buffer binding must be large enough to hold at least one element to avoid
            // triggering validation errors.
            //
            // Note: not using `Ord::max` here because it doesn't support const eval yet (except
            // in nightly)
            len: if len > 0 { len } else { 1 },
            _phantom: std::marker::PhantomData,
        }
    }

    /// Creates a new buffer size from size in bytes.
    pub const fn from_size_in_bytes(size: u32) -> Self {
        Self::new(size / size_of::<T>() as u32)
    }

    /// Returns the number of elements.
    #[expect(clippy::len_without_is_empty, reason = "The buffer can never be empty")]
    pub const fn len(self) -> u32 {
        self.len
    }

    /// Returns the size in bytes.
    pub const fn size_in_bytes(self) -> u32 {
        size_of::<T>() as u32 * self.len
    }

    /// Returns the size in bytes aligned up to the given value.
    pub const fn aligned_in_bytes(self, alignment: u32) -> u32 {
        align_up(self.size_in_bytes(), alignment)
    }
}

impl<T: Sized> PartialEq for BufferSize<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
    }
}

impl<T: Sized> PartialOrd for BufferSize<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.len.partial_cmp(&other.len)
    }
}

/// Computed sizes for all buffers.
#[derive(Copy, Clone, Debug, Default)]
pub struct BufferSizes {
    // Known size buffers
    pub path_reduced: BufferSize<PathMonoid>,
    pub path_reduced2: BufferSize<PathMonoid>,
    pub path_reduced_scan: BufferSize<PathMonoid>,
    pub path_monoids: BufferSize<PathMonoid>,
    pub path_bboxes: BufferSize<PathBbox>,
    pub draw_reduced: BufferSize<DrawMonoid>,
    pub draw_monoids: BufferSize<DrawMonoid>,
    pub info: BufferSize<u32>,
    pub clip_inps: BufferSize<Clip>,
    pub clip_els: BufferSize<ClipElement>,
    pub clip_bics: BufferSize<ClipBic>,
    pub clip_bboxes: BufferSize<ClipBbox>,
    pub draw_bboxes: BufferSize<DrawBbox>,
    pub bump_alloc: BufferSize<BumpAllocators>,
    pub indirect_count: BufferSize<IndirectCount>,
    pub bin_headers: BufferSize<BinHeader>,
    pub paths: BufferSize<Path>,
    // Bump allocated buffers
    pub lines: BufferSize<LineSoup>,
    pub bin_data: BufferSize<u32>,
    pub tiles: BufferSize<Tile>,
    pub seg_counts: BufferSize<SegmentCount>,
    pub segments: BufferSize<PathSegment>,
    pub blend_spill: BufferSize<u32>,
    pub ptcl: BufferSize<u32>,
}

impl BufferSizes {
    pub fn new(layout: &Layout, workgroups: &WorkgroupCounts) -> Self {
        let n_paths = layout.n_paths;
        let n_draw_objects = layout.n_draw_objects;
        let n_clips = layout.n_clips;
        let path_tag_wgs = workgroups.path_reduce.0;
        let reduced_size = if workgroups.use_large_path_scan {
            align_up(path_tag_wgs, PATH_REDUCE_WG)
        } else {
            path_tag_wgs
        };
        let path_reduced = BufferSize::new(reduced_size);
        let path_reduced2 = BufferSize::new(PATH_REDUCE_WG);
        let path_reduced_scan = BufferSize::new(reduced_size);
        let path_monoids = BufferSize::new(path_tag_wgs * PATH_REDUCE_WG);
        let path_bboxes = BufferSize::new(n_paths);
        let binning_wgs = workgroups.binning.0;
        let draw_monoid_wgs = workgroups.draw_reduce.0;
        let draw_reduced = BufferSize::new(draw_monoid_wgs);
        let draw_monoids = BufferSize::new(n_draw_objects);
        let info = BufferSize::new(layout.bin_data_start);
        let clip_inps = BufferSize::new(n_clips);
        let clip_els = BufferSize::new(n_clips);
        let clip_bics = BufferSize::new(n_clips / CLIP_REDUCE_WG);
        let clip_bboxes = BufferSize::new(n_clips);
        let draw_bboxes = BufferSize::new(n_paths);
        let bump_alloc = BufferSize::new(1);
        let indirect_count = BufferSize::new(1);
        let n_paths_aligned = align_up(n_paths, 256);
        let paths = BufferSize::new(n_paths_aligned);
        let width_in_bins = workgroups.coarse.0;
        let height_in_bins = workgroups.coarse.1;
        let n_bins = width_in_bins * height_in_bins;
        let aligned_n_bins = align_up(n_bins, 256);
        let bin_headers = BufferSize::new(binning_wgs * aligned_n_bins);

        // The following buffer sizes have been hand picked to accommodate the vello test scenes as
        // well as paris-30k. These should instead get derived from the scene layout using
        // reasonable heuristics.
        let bin_data = BufferSize::new(1 << 18);
        let tiles = BufferSize::new(1 << 21);
        let lines = BufferSize::new(1 << 21);
        let seg_counts = BufferSize::new(1 << 21);
        let segments = BufferSize::new(1 << 21);
        // 16 * 16 (1 << 8) is one blend spill, so this allows for 4096 spills.
        let blend_spill = BufferSize::new(1 << 20);
        let ptcl = BufferSize::new(1 << 23);
        Self {
            path_reduced,
            path_reduced2,
            path_reduced_scan,
            path_monoids,
            path_bboxes,
            draw_reduced,
            draw_monoids,
            info,
            clip_inps,
            clip_els,
            clip_bics,
            clip_bboxes,
            draw_bboxes,
            bump_alloc,
            indirect_count,
            lines,
            bin_headers,
            paths,
            bin_data,
            tiles,
            seg_counts,
            segments,
            blend_spill,
            ptcl,
        }
    }
}

const fn align_up(len: u32, alignment: u32) -> u32 {
    len + (len.wrapping_neg() & (alignment - 1))
}

#[cfg(test)]
mod bump_sizes_tests {
    use super::*;

    fn config(w: u32, h: u32) -> ConfigUniform {
        RenderConfig::new(&Layout::default(), w, h, &peniko::Color::WHITE).gpu
    }

    /// A config whose bump buffers are sized as a renderer that reads its
    /// counters back sizes them, for an empty scene.
    fn exact_config(w: u32, h: u32) -> ConfigUniform {
        let layout = Layout::default();
        let sizes = BumpSizes::initial(&layout, w, h, u64::MAX);
        RenderConfig::new(&layout, w, h, &peniko::Color::WHITE)
            .with_exact_bump_sizes(&sizes)
            .gpu
    }

    #[test]
    fn exact_sizes_set_vello_s_aside_and_at_least_keeps_them() {
        let small = exact_config(1280, 40);
        let initial = BumpSizes::initial(&Layout::default(), 1280, 40, u64::MAX);
        assert_eq!(BumpSizes::of(&small), initial);
        assert!(small.lines_size < 1 << 21 && small.ptcl_size < 1 << 23);
        // The path without readback keeps Vello's sizes as its least.
        let kept = RenderConfig::new(&Layout::default(), 1280, 40, &peniko::Color::WHITE)
            .with_bump_sizes(&initial)
            .gpu;
        assert_eq!(kept.lines_size, 1 << 21);
        assert_eq!(kept.ptcl_size, 1 << 23);
    }

    #[test]
    fn a_frame_that_ran_out_of_lines_grows_the_segments_after_it() {
        let g = exact_config(1920, 1080);
        let had = BumpSizes::of(&g);
        // Flattening failed; path_count and path_tiling stopped short.
        let bump = BumpAllocators {
            failed: STAGE_FLATTEN,
            lines: 400_000,
            seg_counts: 10,
            segments: 10,
            ..Default::default()
        };
        let grown = BumpSizes::grown_for(&g, &bump, u64::MAX);
        assert_eq!(grown.lines, 500_000);
        // Eight segments a grown line, with the quarter every growth adds.
        assert_eq!(grown.seg_counts, 8 * 500_000 + 8 * 500_000 / 4);
        assert!(grown.segments >= grown.seg_counts);
        assert_eq!(grown.tiles, had.tiles, "tiles do not follow lines");
        assert_eq!(grown.ptcl, had.ptcl);
        // Without a flatten failure the segments only follow what they asked.
        let bump = BumpAllocators {
            failed: STAGE_PATH_COUNT,
            lines: 1000,
            seg_counts: had.seg_counts + 100,
            ..Default::default()
        };
        let grown = BumpSizes::grown_for(&g, &bump, u64::MAX);
        assert_eq!(grown.lines, had.lines);
        let wanted = had.seg_counts + 100;
        assert_eq!(grown.seg_counts, wanted + wanted / 4);
    }

    #[test]
    fn an_overflowed_stage_grows_to_what_it_asked_for_with_room_to_spare() {
        let g = config(3840, 2160);
        let had = BumpSizes::of(&g);
        let bump = BumpAllocators {
            failed: STAGE_TILE_ALLOC,
            tile: 6_645_984,
            lines: 1600,
            ..Default::default()
        };
        let grown = BumpSizes::grown_for(&g, &bump, u64::MAX);
        assert_eq!(grown.tiles, 6_645_984 + 6_645_984 / 4);
        assert_eq!(
            grown.lines, had.lines,
            "a stage that did not fail keeps its size"
        );
        assert_eq!(grown.binning, had.binning);
        // A config with the grown sizes says so to the shaders.
        let mut rc = RenderConfig::new(&Layout::default(), 3840, 2160, &peniko::Color::WHITE);
        rc = rc.with_bump_sizes(&grown);
        assert_eq!(rc.gpu.tiles_size, grown.tiles);
        assert_eq!(rc.buffer_sizes.tiles.len(), grown.tiles);
        assert_eq!(BumpSizes::of(&rc.gpu), grown);
    }

    #[test]
    fn floors_leave_vello_s_sizes_up_to_1440p() {
        for (w, h) in [(800, 600), (1920, 1080), (2560, 1440)] {
            let plain = RenderConfig::new(&Layout::default(), w, h, &peniko::Color::WHITE);
            let had = BumpSizes::of(&plain.gpu);
            let floored = plain.with_bump_sizes(&BumpSizes::floors(w, h, 128 << 20));
            assert_eq!(BumpSizes::of(&floored.gpu), had, "{w}x{h}");
        }
    }

    #[test]
    fn floors_follow_the_target_past_vello_s_sizes() {
        // 4K is 240x135 tiles: the tiles keep Vello's size, the words pass it.
        let floors = BumpSizes::floors(3840, 2160, 128 << 20);
        assert_eq!(floors.ptcl, 240 * 135 * 512);
        let rc = RenderConfig::new(&Layout::default(), 3840, 2160, &peniko::Color::WHITE);
        let had = BumpSizes::of(&rc.gpu);
        let rc = rc.with_bump_sizes(&floors);
        assert_eq!(rc.gpu.ptcl_size, floors.ptcl);
        assert_eq!(rc.gpu.tiles_size, had.tiles);
        // 8K is 480x270 tiles.
        let floors = BumpSizes::floors(7680, 4320, u64::MAX);
        assert_eq!(floors.tiles, 480 * 270 * 32);
        assert_eq!(floors.ptcl, 480 * 270 * 512);
        let rc = RenderConfig::new(&Layout::default(), 7680, 4320, &peniko::Color::WHITE);
        let had = BumpSizes::of(&rc.gpu);
        let rc = rc.with_bump_sizes(&floors);
        assert_eq!(rc.gpu.tiles_size, floors.tiles);
        assert_eq!(rc.buffer_sizes.ptcl.len(), floors.ptcl);
        // The floors are of these two buffers alone.
        assert_eq!(rc.gpu.lines_size, had.lines);
        assert_eq!(rc.gpu.segments_size, had.segments);
        assert_eq!(rc.gpu.blend_size, had.blend);
        // At wgpu's default limit the 8K words stop at 128 MiB.
        assert_eq!(
            BumpSizes::floors(7680, 4320, 128 << 20).ptcl as u64 * 4,
            128 << 20
        );
        // A partly covered tile on the edge of the target counts as a tile.
        assert_eq!(BumpSizes::floors(161, 17, u64::MAX).tiles, 11 * 2 * 32);
    }

    #[test]
    fn floors_stay_under_the_limit_once_the_pool_rounds_them_up() {
        // wgpu's default limit is a size class: taken as it is.
        let ptcl = BumpSizes::floors(16384, 16384, 128 << 20).ptcl as u64;
        assert_eq!(ptcl * 4, 128 << 20);
        // Between classes, the floor falls to the class under it.
        let floors = BumpSizes::floors(16384, 16384, (128 << 20) + 1);
        assert_eq!(floors.ptcl as u64 * 4, 128 << 20);
        let floors = BumpSizes::floors(65536, 65536, (1 << 31) - 1);
        assert_eq!(floors.ptcl as u64 * 4, 3 << 29);
        assert_eq!(floors.tiles as u64 * size_of::<Tile>() as u64, 3 << 29);
        // A target too wide for the arithmetic saturates to the same cap.
        let floors = BumpSizes::floors(u32::MAX, u32::MAX, 128 << 20);
        assert_eq!(floors.ptcl as u64 * 4, 128 << 20);
    }

    #[test]
    fn a_grown_buffer_stays_under_the_limit_once_the_pool_rounds_it_up() {
        assert_eq!(pool_size_class_below(128 << 20), 128 << 20);
        assert_eq!(pool_size_class_below(256 << 20), 256 << 20);
        assert_eq!(pool_size_class_below((1 << 31) - 1), 3 << 29);
        assert_eq!(pool_size_class_below((3 << 29) - 1), 1 << 30);

        let g = config(3840, 2160);
        let bump = BumpAllocators {
            failed: STAGE_TILE_ALLOC,
            tile: u32::MAX / 16,
            ..Default::default()
        };
        let tiles = BumpSizes::grown_for(&g, &bump, (1 << 31) - 1).tiles as u64;
        assert_eq!(tiles * size_of::<Tile>() as u64, 3 << 29);
    }

    #[test]
    fn a_stage_that_failed_without_a_count_doubles_and_nothing_passes_the_device_limit() {
        let g = config(1920, 1080);
        let had = BumpSizes::of(&g);
        // Coarse gave up: ptcl counted only what it reached.
        let bump = BumpAllocators {
            failed: STAGE_COARSE | STAGE_PATH_COUNT,
            ..Default::default()
        };
        let grown = BumpSizes::grown_for(&g, &bump, u64::MAX);
        assert_eq!(grown.ptcl, had.ptcl * 2);
        assert_eq!(grown.seg_counts, had.seg_counts * 2);
        assert!(
            grown.segments >= grown.seg_counts,
            "a segment per counted segment"
        );
        // At the device's limit nothing can grow: the sizes come back unchanged.
        let limit = had.tiles as u64 * size_of::<Tile>() as u64;
        let bump = BumpAllocators {
            failed: STAGE_TILE_ALLOC,
            tile: had.tiles * 3,
            ..Default::default()
        };
        assert_eq!(BumpSizes::grown_for(&g, &bump, limit).tiles, had.tiles);
        // The limit holds `bin_data` whole, its info included.
        let limit = (g.layout.bin_data_start as u64 + had.binning as u64 * 2) * 4;
        let bump = BumpAllocators {
            failed: STAGE_BINNING,
            binning: had.binning * 3,
            ..Default::default()
        };
        assert_eq!(
            BumpSizes::grown_for(&g, &bump, limit).binning,
            had.binning * 2
        );
        // The sizes count bytes in a `u32`, whatever the device allows.
        let bump = BumpAllocators {
            failed: STAGE_TILE_ALLOC,
            tile: u32::MAX,
            ..Default::default()
        };
        let tiles = BumpSizes::grown_for(&g, &bump, u64::MAX).tiles as u64;
        assert!(tiles * size_of::<Tile>() as u64 <= u32::MAX as u64);
        // A frame that did not fail changes nothing.
        assert_eq!(
            BumpSizes::grown_for(&g, &BumpAllocators::default(), u64::MAX),
            had
        );
    }
}
