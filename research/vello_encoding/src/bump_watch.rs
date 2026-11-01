// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! When a renderer has to wait for a frame's allocation counters.
//!
//! Reading [`BumpAllocators`] back as soon as a frame is submitted makes the
//! CPU wait for the GPU, and on a shared device for everything else on it.
//! That wait buys one thing: a frame that overflowed can be drawn again before
//! it is shown. [`BumpWatch`] keeps it for the frames that may need it, and
//! lets the rest collect their counters later.

use crate::{BumpAllocators, BumpSizes, ConfigUniform};

/// What a frame's target and resolved scene measure, taken from its config:
/// the cheap numbers its allocation grows with. These come from the resolved
/// layout rather than the [`Encoding`](crate::Encoding), whose counts leave out
/// the glyphs of its text.
///
/// None of them sees how much of the target a path covers, so a scene can use
/// more tiles without measuring any larger; see [`BumpWatch`] for what catches
/// that.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FrameShape {
    pub width: u32,
    pub height: u32,
    pub n_paths: u32,
    pub n_clips: u32,
    pub n_draw_objects: u32,
    /// Words of path data, which stands in for the number of segments.
    pub path_data: u32,
}

impl FrameShape {
    /// The shape of a frame recorded with config `g`.
    pub fn of(g: &ConfigUniform) -> Self {
        let layout = &g.layout;
        Self {
            width: g.target_width,
            height: g.target_height,
            n_paths: layout.n_paths,
            n_clips: layout.n_clips,
            n_draw_objects: layout.n_draw_objects,
            path_data: layout.draw_tag_base.saturating_sub(layout.path_data_base),
        }
    }

    fn at_least(self, other: Self) -> Self {
        Self {
            width: self.width.max(other.width),
            height: self.height.max(other.height),
            n_paths: self.n_paths.max(other.n_paths),
            n_clips: self.n_clips.max(other.n_clips),
            n_draw_objects: self.n_draw_objects.max(other.n_draw_objects),
            path_data: self.path_data.max(other.path_data),
        }
    }

    /// Whether this frame leaves what `verified` covers: any larger on the
    /// target, or more than a quarter over it on any scene measure.
    fn exceeds(&self, verified: &Self) -> bool {
        let over = |now: u32, high: u32| now > high.saturating_add(high / 4);
        self.width > verified.width
            || self.height > verified.height
            || over(self.n_paths, verified.n_paths)
            || over(self.n_clips, verified.n_clips)
            || over(self.n_draw_objects, verified.n_draw_objects)
            || over(self.path_data, verified.path_data)
    }
}

/// Decides which frames are at risk of overflowing, from what earlier frames'
/// counters said, however late those arrived. A frame is at risk when:
///
/// - no frame has been verified yet, as after a renderer is created;
/// - its [`FrameShape`] exceeds the highs of the frames that came back without
///   overflowing (the highs are per measure and never fall, so scenes taking
///   turns on one renderer do not put each other at risk); or
/// - the last counters seen were a new high at more than 3/4 of a buffer, or
///   an overflow.
///
/// The third rule asks for a *new* high on purpose. A buffer grown for a frame
/// holds it at 4/5 full ([`BumpSizes::grown_for`]) and stays that way, so
/// waiting whenever usage is near a limit would wait on every later frame. It
/// is also what catches usage that grows without the shape growing, like a path
/// scaled up over the target, as long as it climbs through the last quarter.
#[derive(Clone, Debug, Default)]
pub struct BumpWatch {
    verified: Option<FrameShape>,
    used_high: BumpSizes,
    armed: bool,
}

impl BumpWatch {
    /// Whether a frame of `shape` is at risk, so that the caller should wait
    /// for its counters. This consumes the third rule: the counters of the
    /// frame that waits arm it again if they need to.
    pub fn take_risk(&mut self, shape: &FrameShape) -> bool {
        let armed = core::mem::take(&mut self.armed);
        armed || self.verified.is_none_or(|v| shape.exceeds(&v))
    }

    /// Takes in the counters `bump` of a frame recorded with config `g`, at
    /// any time after it: `sizes` are the renderer's sizes now, which may have
    /// passed the ones that frame had. Returns the sizes to keep, which never
    /// fall below `sizes` and pass them only when the frame overflowed
    /// something they still do not cover. `max_bytes` is as for
    /// [`BumpSizes::grown_for`].
    pub fn observe(
        &mut self,
        g: &ConfigUniform,
        bump: &BumpAllocators,
        sizes: BumpSizes,
        max_bytes: u64,
    ) -> BumpSizes {
        let sizes = sizes.at_least(BumpSizes::grown_for(g, bump, max_bytes));
        let used = BumpSizes::used(g, bump);
        let near_new_high = used.nears_limit(&sizes, &self.used_high);
        self.used_high = self.used_high.at_least(used);
        self.armed |= bump.failed != 0 || near_new_high;
        if bump.failed == 0 {
            let shape = FrameShape::of(g);
            self.verified = Some(self.verified.map_or(shape, |v| v.at_least(shape)));
        }
        sizes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Layout, RenderConfig};

    const NO_LIMIT: u64 = u64::MAX;

    fn config(width: u32, height: u32, n_paths: u32) -> ConfigUniform {
        let layout = Layout {
            n_paths,
            n_draw_objects: n_paths,
            n_clips: n_paths / 10,
            path_data_base: 100,
            draw_tag_base: 100 + n_paths * 8,
            ..Layout::default()
        };
        RenderConfig::new(&layout, width, height, &peniko::Color::WHITE).gpu
    }

    fn fits() -> BumpAllocators {
        BumpAllocators {
            tile: 1_000,
            ..BumpAllocators::default()
        }
    }

    /// A watch that has verified a 800x600 frame of 100 paths, and is calm.
    fn verified() -> (BumpWatch, ConfigUniform) {
        let g = config(800, 600, 100);
        let mut watch = BumpWatch::default();
        watch.observe(&g, &fits(), BumpSizes::of(&g), NO_LIMIT);
        assert!(!watch.take_risk(&FrameShape::of(&g)));
        (watch, g)
    }

    #[test]
    fn the_first_frame_is_at_risk_until_one_is_verified() {
        let g = config(800, 600, 100);
        let shape = FrameShape::of(&g);
        let mut watch = BumpWatch::default();
        assert!(watch.take_risk(&shape));
        assert!(watch.take_risk(&shape), "asking verifies nothing");
        let failed = BumpAllocators {
            failed: 0x2,
            tile: u32::MAX / 2,
            ..BumpAllocators::default()
        };
        watch.observe(&g, &failed, BumpSizes::of(&g), NO_LIMIT);
        watch.take_risk(&shape);
        assert!(watch.take_risk(&shape), "an overflow verifies nothing");
        watch.observe(&g, &fits(), BumpSizes::of(&g), NO_LIMIT);
        assert!(!watch.take_risk(&shape));
    }

    #[test]
    fn any_growth_of_the_target_is_at_risk_and_no_shrink_is() {
        let (mut watch, g) = verified();
        for (w, h) in [(801, 600), (800, 601), (801, 1)] {
            assert!(watch.take_risk(&FrameShape::of(&config(w, h, 100))));
        }
        for (w, h) in [(800, 600), (799, 600), (1, 1)] {
            assert!(!watch.take_risk(&FrameShape::of(&config(w, h, 100))));
        }
        // Width and height are held apart: a tall frame and a wide one do not
        // verify a frame as large as both.
        let tall = config(100, 2_000, 100);
        watch.observe(&tall, &fits(), BumpSizes::of(&tall), NO_LIMIT);
        assert!(!watch.take_risk(&FrameShape::of(&g)));
        assert!(watch.take_risk(&FrameShape::of(&config(801, 2_000, 100))));
    }

    #[test]
    fn each_scene_measure_has_a_quarter_of_margin() {
        let (mut watch, g) = verified();
        let high = FrameShape::of(&g);
        let in_margin = FrameShape {
            n_paths: 125,
            n_clips: 12,
            n_draw_objects: 125,
            path_data: 1_000,
            ..high
        };
        assert!(!watch.take_risk(&in_margin));
        let past: [fn(&mut FrameShape); 4] = [
            |s| s.n_paths += 1,
            |s| s.n_clips += 1,
            |s| s.n_draw_objects += 1,
            |s| s.path_data += 1,
        ];
        for step in past {
            let mut shape = in_margin;
            step(&mut shape);
            assert!(watch.take_risk(&shape), "{shape:?}");
        }
    }

    #[test]
    fn scenes_taking_turns_do_not_put_each_other_at_risk() {
        let (mut watch, small) = verified();
        let big = config(800, 600, 10_000);
        assert!(watch.take_risk(&FrameShape::of(&big)));
        watch.observe(&big, &fits(), BumpSizes::of(&big), NO_LIMIT);
        for _ in 0..3 {
            assert!(!watch.take_risk(&FrameShape::of(&small)));
            assert!(!watch.take_risk(&FrameShape::of(&big)));
        }
    }

    #[test]
    fn a_late_verified_frame_raises_the_highs() {
        let (mut watch, g) = verified();
        let later = config(800, 600, 125);
        assert!(!watch.take_risk(&FrameShape::of(&later)));
        watch.observe(&later, &fits(), BumpSizes::of(&later), NO_LIMIT);
        assert!(!watch.take_risk(&FrameShape::of(&config(800, 600, 156))));
        assert!(watch.take_risk(&FrameShape::of(&config(800, 600, 157))));
        // A verified high outlives a smaller result arriving after it.
        watch.observe(&g, &fits(), BumpSizes::of(&g), NO_LIMIT);
        assert!(!watch.take_risk(&FrameShape::of(&config(800, 600, 156))));
    }

    #[test]
    fn a_late_overflow_grows_the_sizes_and_puts_the_next_frame_at_risk() {
        let (mut watch, g) = verified();
        let sizes = BumpSizes::of(&g);
        let overflowed = BumpAllocators {
            failed: 0x2,
            tile: sizes.tiles + 1_000,
            ..BumpAllocators::default()
        };
        let grown = watch.observe(&g, &overflowed, sizes, NO_LIMIT);
        assert_eq!(grown, BumpSizes::grown_for(&g, &overflowed, NO_LIMIT));
        assert!(grown.tiles > sizes.tiles + 1_000);
        assert!(watch.take_risk(&FrameShape::of(&g)));
        assert!(!watch.take_risk(&FrameShape::of(&g)), "only that frame");
    }

    #[test]
    fn a_late_overflow_never_undoes_a_later_growth() {
        let (mut watch, g) = verified();
        let had = BumpSizes::of(&g);
        // Grown by a frame that waited, while this one was on its way back.
        let sizes = BumpSizes {
            ptcl: had.ptcl * 4,
            tiles: had.tiles * 4,
            ..had
        };
        let overflowed = BumpAllocators {
            failed: 0x2,
            tile: had.tiles + 1_000,
            lines: had.lines + 1_000,
            ..BumpAllocators::default()
        };
        let kept = watch.observe(&g, &overflowed, sizes, NO_LIMIT);
        assert_eq!(kept.ptcl, sizes.ptcl);
        assert_eq!(kept.tiles, sizes.tiles, "already covers what it wanted");
        assert!(kept.lines > had.lines, "still grows what it does not cover");
        let fine = watch.observe(&g, &fits(), kept, NO_LIMIT);
        assert_eq!(fine, kept);
    }

    #[test]
    fn near_a_limit_counts_only_as_a_new_high() {
        let (mut watch, g) = verified();
        let sizes = BumpSizes::of(&g);
        let at = |tile: u32| BumpAllocators {
            tile,
            ..BumpAllocators::default()
        };
        let shape = FrameShape::of(&g);
        // Up to 3/4 is calm, whatever it does.
        watch.observe(&g, &at(sizes.tiles / 4 * 3), sizes, NO_LIMIT);
        assert!(!watch.take_risk(&shape));
        let near = sizes.tiles / 4 * 3 + 1;
        watch.observe(&g, &at(near), sizes, NO_LIMIT);
        assert!(watch.take_risk(&shape));
        // Holding there, or falling back, does not wait again.
        watch.observe(&g, &at(near), sizes, NO_LIMIT);
        watch.observe(&g, &at(near - 1), sizes, NO_LIMIT);
        assert!(!watch.take_risk(&shape));
        watch.observe(&g, &at(near + 1), sizes, NO_LIMIT);
        assert!(watch.take_risk(&shape));
    }

    /// The case the "new high" exists for: after an overflow, a frame the
    /// growth just fitted sits over 3/4 of the buffer for good.
    #[test]
    fn a_frame_steady_in_a_grown_buffer_stops_waiting() {
        let g = config(800, 600, 100);
        let shape = FrameShape::of(&g);
        let mut watch = BumpWatch::default();
        let wanted = BumpSizes::of(&g).tiles * 2;
        let overflowed = BumpAllocators {
            failed: 0x2,
            tile: wanted,
            ..BumpAllocators::default()
        };
        assert!(watch.take_risk(&shape));
        let grown = watch.observe(&g, &overflowed, BumpSizes::of(&g), NO_LIMIT);
        let redrawn = BumpAllocators {
            tile: wanted,
            ..BumpAllocators::default()
        };
        assert!(BumpSizes::used(&g, &redrawn).nears_limit(&grown, &BumpSizes::default()));
        assert_eq!(watch.observe(&g, &redrawn, grown, NO_LIMIT), grown);
        assert!(watch.take_risk(&shape), "the overflow itself still counts");
        assert_eq!(watch.observe(&g, &redrawn, grown, NO_LIMIT), grown);
        for _ in 0..3 {
            assert!(!watch.take_risk(&shape));
            watch.observe(&g, &redrawn, grown, NO_LIMIT);
        }
    }

    #[test]
    fn the_ptcl_limit_is_held_against_the_static_part_too() {
        let (mut watch, g) = verified();
        let sizes = BumpSizes::of(&g);
        let ptcl_static = BumpSizes::used(&g, &BumpAllocators::default()).ptcl;
        assert!(ptcl_static > 0);
        let near = BumpAllocators {
            ptcl: sizes.ptcl / 4 * 3 + 1 - ptcl_static,
            ..BumpAllocators::default()
        };
        watch.observe(&g, &near, sizes, NO_LIMIT);
        assert!(watch.take_risk(&FrameShape::of(&g)));
    }
}
