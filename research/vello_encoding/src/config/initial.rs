// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The bump buffers' sizes for a frame before anything is known of what it
//! needs: derived from its target and its resolved scene, not fixed.
//!
//! `BufferSizes::new`'s constants were picked for Vello's test scenes and
//! paris-30k and do not change with what is drawn: every renderer held about
//! 165 MiB of them, a 1280x40 bar included (rinch #2466). Where a frame's
//! counters are read back, an overflow is found and grown from (see
//! [`BumpSizes::grown_for`]), so the first sizes can follow the frame instead.
//!
//! Each rule below was calibrated against the counts of real frames (rinch
//! #2466): rinch's examples and a deliberately curve-heavy scene, at 1280x720,
//! 1920x1080 and 3840x2160, on a Radeon 890M. The margin a rule leaves over
//! the most any recorded frame needed is stated with it, and a test pins it
//! against those recorded counts.

use super::{BumpSizes, TILE_HEIGHT, TILE_WIDTH, pool_size_class_below};
use crate::{Layout, LineSoup, PathSegment, SegmentCount, Tile};

/// Tiles per target tile per path, at most this many paths: `tile_alloc`
/// gives each path its bounding box in tiles, at most the whole target, so
/// `paths x target tiles` bounds it exactly; past this many paths the bound
/// stops being useful and this rule takes over. The most measured was 30.7
/// (a window of overlapping floating panels), so the margin is 2.1.
const TILES_PER_TARGET_TILE: u32 = 64;
const TILES_FLOOR: u32 = 1 << 12;

/// Per-tile command list words per target tile, the static 64 included.
/// Kept from the floor before (vello-fork #4: 8 full-window opacity layers at
/// 4K overflow 256). The most measured was 110, so the margin is 4.6.
const PTCL_WORDS_PER_TARGET_TILE: u32 = 512;

/// Lines per path tag: flattening turns a segment into at least one line,
/// a curve drawn large into several. The most measured was 3.6 (circles of up
/// to 340 px radius, wide round strokes, at every size; a curve's line count
/// grows with the square root of its scale), so the margin is 2.2.
const LINES_PER_PATH_TAG: u32 = 8;
const LINES_FLOOR: u32 = 1 << 14;

/// Segments (and segment counts) per path tag, plus per target tile: a line
/// is cut at every tile it crosses, so long and large shapes cross more as
/// the target grows (the curve scene: 6.1, 6.8 and 8.2 per tag at 720p,
/// 1080p and 4K). The margin over the most measured is 3.1.
const SEGMENTS_PER_PATH_TAG: u32 = 16;
const SEGMENTS_PER_TARGET_TILE: u32 = 4;
const SEGMENTS_FLOOR: u32 = 1 << 15;

/// Binning: each draw object is binned at most once per 256x256 bin, so
/// `draw objects x bins` bounds it exactly; capped at Vello's own size, so
/// never more than before.
const BINNING_FLOOR: u32 = 1 << 10;
const BINNING_CAP: u32 = 1 << 18;

/// The clips and layers the fine stage holds per pixel before it spills
/// (`BLEND_STACK_SPLIT` in `shader/shared/config.wgsl`). `coarse.wgsl`
/// allocates blend spill for a tile only `if max_blend_depth >
/// BLEND_STACK_SPLIT`, and a tile's blend depth is at most the scene's
/// deepest nesting, so a scene nested this deep or less spills nothing.
const BLEND_STACK_SPLIT: u32 = 4;

/// Blend spill words a tile takes per layer past the stack: `coarse` gives a
/// tile `2 x (max_blend_depth - BLEND_STACK_SPLIT) x 16 x 16` words (2 a
/// pixel, the 16-bit backdrop of vello-fork #5). For every tile of the target
/// at the scene's deepest nesting, that bounds the buffer exactly.
const BLEND_WORDS_PER_SPILLED_LAYER: u32 = 2 * TILE_WIDTH * TILE_HEIGHT;

/// The bound is capped at Vello's own size, `BufferSizes::new`'s 2^20 words:
/// a scene that fits that today still fits, and one that outgrows it
/// overflows once and grows, as before. (Sizing every tile at the deepest
/// nesting would cost a 4K window about 63 MiB per layer past the stack,
/// where a real scene spills in few tiles: Pimble used 128 000 words at 4K,
/// vello-fork #3.)
const BLEND_CAP: u32 = 1 << 20;
const BLEND_FLOOR: u32 = 1 << 12;

/// The size of a bin, in tiles, on each axis.
const BIN_TILES: u32 = 16;

impl BumpSizes {
    /// The sizes a frame of a `width`x`height` target drawing the resolved
    /// scene `layout`, whose clips and layers nest `clip_depth` deep at most
    /// ([`Encoding::max_open_clips`](crate::Encoding::max_open_clips)),
    /// starts with, before its counters say what it needed.
    ///
    /// No buffer passes `max_bytes`, the device's largest storage binding,
    /// nor does the engine's pool take one past it by rounding it up.
    ///
    /// The path tag count stands for the scene's path segments: it is a few
    /// more (each path has a marker tag), and unlike the encoding's own count
    /// it includes the outlines of glyph runs, which are only added when the
    /// scene is resolved.
    pub fn initial(
        layout: &Layout,
        clip_depth: u32,
        width: u32,
        height: u32,
        max_bytes: u64,
    ) -> Self {
        let width_in_tiles = width.div_ceil(TILE_WIDTH);
        let height_in_tiles = height.div_ceil(TILE_HEIGHT);
        let tiles = width_in_tiles.saturating_mul(height_in_tiles);
        let bins = width_in_tiles
            .div_ceil(BIN_TILES)
            .saturating_mul(height_in_tiles.div_ceil(BIN_TILES));
        let tags = layout.path_tags_size();
        let max_bytes = pool_size_class_below(max_bytes.min(u32::MAX as u64));
        let cap = |len: u32, elem_bytes: usize| len.min((max_bytes / elem_bytes as u64) as u32);
        let segments = SEGMENTS_PER_PATH_TAG
            .saturating_mul(tags)
            .saturating_add(SEGMENTS_PER_TARGET_TILE.saturating_mul(tiles))
            .max(SEGMENTS_FLOOR);
        Self {
            lines: cap(
                LINES_PER_PATH_TAG.saturating_mul(tags).max(LINES_FLOOR),
                size_of::<LineSoup>(),
            ),
            binning: cap(
                layout
                    .n_draw_objects
                    .saturating_mul(bins)
                    .clamp(BINNING_FLOOR, BINNING_CAP),
                size_of::<u32>(),
            )
            .min(
                ((max_bytes / size_of::<u32>() as u64) as u32)
                    .saturating_sub(layout.bin_data_start),
            ),
            tiles: cap(
                layout
                    .n_paths
                    .min(TILES_PER_TARGET_TILE)
                    .saturating_mul(tiles)
                    .max(TILES_FLOOR),
                size_of::<Tile>(),
            ),
            seg_counts: cap(segments, size_of::<SegmentCount>()),
            segments: cap(segments, size_of::<PathSegment>()),
            blend: cap(
                clip_depth
                    .saturating_sub(BLEND_STACK_SPLIT)
                    .saturating_mul(BLEND_WORDS_PER_SPILLED_LAYER)
                    .saturating_mul(tiles)
                    .clamp(BLEND_FLOOR, BLEND_CAP),
                size_of::<u32>(),
            ),
            ptcl: cap(
                PTCL_WORDS_PER_TARGET_TILE.saturating_mul(tiles),
                size_of::<u32>(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A resolved scene with `tags` path tags, `paths` paths and as many draw
    /// objects, and no binning info before the bins.
    fn layout(tags: u32, paths: u32) -> Layout {
        Layout {
            n_paths: paths,
            n_draw_objects: paths,
            path_tag_base: 0,
            path_data_base: tags.div_ceil(4),
            ..Layout::default()
        }
    }

    #[test]
    fn each_rule_s_multiplier_and_floor() {
        // 1280x720: 80 x 45 = 3600 target tiles, 5 x 3 = 15 bins.
        let s = BumpSizes::initial(&layout(4000, 100), 0, 1280, 720, u64::MAX);
        assert_eq!(s.tiles, 64 * 3600);
        assert_eq!(s.ptcl, 512 * 3600);
        assert_eq!(s.lines, 8 * 4000);
        assert_eq!(s.segments, 16 * 4000 + 4 * 3600);
        assert_eq!(s.seg_counts, s.segments);
        assert_eq!(s.binning, 100 * 15);
        assert_eq!(
            s.blend,
            1 << 12,
            "nested no deeper than the stack: nothing spills"
        );
        // Fewer paths than the tile rule's limit: the exact bound.
        let s = BumpSizes::initial(&layout(4000, 20), 0, 1280, 720, u64::MAX);
        assert_eq!(s.tiles, 20 * 3600);
        // The floors, for an empty scene on a tiny target.
        let s = BumpSizes::initial(&layout(0, 0), 0, 16, 16, u64::MAX);
        assert_eq!(s.tiles, 1 << 12);
        assert_eq!(s.lines, 1 << 14);
        assert_eq!(s.segments, 1 << 15);
        assert_eq!(s.binning, 1 << 10);
        assert_eq!(s.blend, 1 << 12);
        assert_eq!(s.ptcl, 512);
        // Binning never more than Vello's own size.
        let s = BumpSizes::initial(&layout(4000, 100_000), 0, 3840, 2160, u64::MAX);
        assert_eq!(s.binning, 1 << 18);
    }

    #[test]
    fn blend_spill_is_the_exact_bound_capped_at_vello_s_size() {
        // Vello's deep_blend snapshot: 7 nested layers in 200x200, 13 x 13
        // tiles, each spilling 3 layers of 512 words.
        let s = BumpSizes::initial(&layout(40, 16), 7, 200, 200, u64::MAX);
        assert_eq!(s.blend, 3 * 512 * 169);
        // At the stack's depth nothing can spill: the floor alone, at any size.
        for depth in 0..=4 {
            let s = BumpSizes::initial(&layout(40, 16), depth, 3840, 2160, u64::MAX);
            assert_eq!(s.blend, 1 << 12, "depth {depth}");
        }
        // Past it the bound, never under the floor and never over 2^20:
        // one layer past at 4K would be 512 x 32 400 words.
        let s = BumpSizes::initial(&layout(40, 16), 5, 3840, 2160, u64::MAX);
        assert_eq!(s.blend, 1 << 20);
        let s = BumpSizes::initial(&layout(40, 16), 5, 16, 16, u64::MAX);
        assert_eq!(
            s.blend,
            1 << 12,
            "512 words for one tile is under the floor"
        );
    }

    /// The bytes of the buffers a set of sizes asks for.
    fn bytes(s: &BumpSizes) -> u64 {
        s.lines as u64 * size_of::<LineSoup>() as u64
            + s.binning as u64 * 4
            + s.tiles as u64 * size_of::<Tile>() as u64
            + s.seg_counts as u64 * size_of::<SegmentCount>() as u64
            + s.segments as u64 * size_of::<PathSegment>() as u64
            + s.blend as u64 * 4
            + s.ptcl as u64 * 4
    }

    #[test]
    fn a_bar_holds_a_few_mib_not_165() {
        // rinch's wayland-bar: 1280x32, 302 segments and 48 paths.
        let s = BumpSizes::initial(&layout(302 + 48, 48), 0, 1280, 32, 128 << 20);
        assert!(bytes(&s) < 2 << 20, "{} bytes", bytes(&s));
        // A 1280x40 panel.
        let s = BumpSizes::initial(&layout(400, 40), 0, 1280, 40, 128 << 20);
        assert!(bytes(&s) < 2 << 20, "{} bytes", bytes(&s));
    }

    #[test]
    fn no_buffer_passes_the_device_limit() {
        let limit = 128 << 20;
        let s = BumpSizes::initial(&layout(u32::MAX - 3, 1 << 20), 0, 16384, 16384, limit);
        for (len, elem) in [
            (s.lines, size_of::<LineSoup>()),
            (s.tiles, size_of::<Tile>()),
            (s.segments, size_of::<PathSegment>()),
            (s.seg_counts, size_of::<SegmentCount>()),
            (s.ptcl, 4),
            (s.blend, 4),
            (s.binning, 4),
        ] {
            assert!(len as u64 * elem as u64 <= limit, "{len} x {elem}");
        }
    }

    /// What a frame recorded with stock sizes needed (rinch #2466's
    /// calibration, Radeon 890M): the target, the scene's path segments,
    /// paths and draw objects, and its counters.
    struct Recorded {
        width: u32,
        height: u32,
        segments: u32,
        paths: u32,
        lines: u32,
        binning: u32,
        tiles: u32,
        seg_counts: u32,
        ptcl_dynamic: u32,
    }

    /// The curve-heavy scene at each size, and the window of floating panels
    /// that needs the most tiles per target tile.
    const RECORDED: [Recorded; 4] = [
        Recorded {
            width: 1280,
            height: 720,
            segments: 4347,
            paths: 69,
            lines: 15600,
            binning: 631,
            tiles: 102_749,
            seg_counts: 26_687,
            ptcl_dynamic: 12_032,
        },
        Recorded {
            width: 1920,
            height: 1080,
            segments: 5279,
            paths: 69,
            lines: 19_008,
            binning: 1394,
            tiles: 228_354,
            seg_counts: 35_836,
            ptcl_dynamic: 15_360,
        },
        Recorded {
            width: 3840,
            height: 2160,
            segments: 7430,
            paths: 69,
            lines: 26_867,
            binning: 4315,
            tiles: 893_575,
            seg_counts: 61_121,
            ptcl_dynamic: 17_152,
        },
        Recorded {
            width: 1200,
            height: 692,
            segments: 9902,
            paths: 1084,
            lines: 14_301,
            binning: 1856,
            tiles: 101_213,
            seg_counts: 29_396,
            ptcl_dynamic: 151_552,
        },
    ];

    #[test]
    fn recorded_frames_fit_with_the_stated_margins() {
        for r in &RECORDED {
            // The encoding's segment count, which is at most the path tags
            // the rules read: this is the stricter test.
            let s =
                BumpSizes::initial(&layout(r.segments, r.paths), 0, r.width, r.height, u64::MAX);
            let tiles = r.width.div_ceil(16) * r.height.div_ceil(16);
            let fits = |have: u32, need: u32, margin: f64, what: &str| {
                assert!(
                    have as f64 >= need as f64 * margin,
                    "{}x{} {what}: have {have}, need {need} with a margin of {margin}",
                    r.width,
                    r.height
                );
            };
            fits(s.tiles, r.tiles, 2.0, "tiles");
            fits(s.ptcl, 64 * tiles + r.ptcl_dynamic, 4.5, "ptcl");
            fits(s.lines, r.lines, 2.2, "lines");
            fits(s.segments, r.seg_counts, 3.0, "segments");
            fits(s.seg_counts, r.seg_counts, 3.0, "seg_counts");
            fits(s.binning, r.binning, 1.0, "binning");
        }
    }
}
