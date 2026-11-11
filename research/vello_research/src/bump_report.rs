// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Reports of what frames' bump-allocated buffers needed, for measuring the
//! rules that size them ([`BumpSizes::initial`]).
//!
//! The robust path reads every frame's counters back already, to grow the
//! buffers when one ran out. A [`Renderer`](crate::Renderer) asked to with
//! [`record_bump_reports`](crate::Renderer::record_bump_reports) also keeps
//! what each frame needed, what it was given and what its initial sizes were
//! derived from, until [`take_bump_reports`](crate::Renderer::take_bump_reports)
//! drains them. Not asked, it keeps nothing.

use std::collections::VecDeque;
use vello_encoding::{BumpAllocators, BumpSizes, ConfigUniform};

/// What one frame's bump-allocated buffers needed, beside what the frame was
/// given and what [`BumpSizes::initial`] derives its sizes from: what a
/// recalibration of those rules reads.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct BumpReport {
    /// The counters the GPU wrote: how many elements each buffer was asked
    /// for, and in `failed` the stages that ran out. [`Self::needed`] says it
    /// per buffer.
    pub counters: BumpAllocators,
    /// The sizes the frame was given.
    pub given: BumpSizes,
    /// The frame's configuration: its target size (`target_width`,
    /// `target_height`) and its scene's layout (path tags, paths, draw
    /// objects, clips), which [`BumpSizes::initial`] reads.
    pub config: ConfigUniform,
    /// The scene's deepest nesting of clips and layers
    /// (`Encoding::max_open_clips`), the rest of what it reads.
    pub max_open_clips: u32,
    /// The device's largest buffer the sizes were capped at, in bytes.
    pub max_bytes: u64,
}

impl BumpReport {
    /// What the frame asked of each buffer, in elements, comparable with
    /// [`Self::given`] and [`Self::initial`] ([`BumpSizes::needed`]).
    pub fn needed(&self) -> BumpSizes {
        BumpSizes::needed(&self.config, &self.counters)
    }

    /// The first sizes the rules give this frame's target and scene
    /// ([`BumpSizes::initial`]), before the renderer's high-water mark: what
    /// [`Self::needed`] is measured against to calibrate them.
    pub fn initial(&self) -> BumpSizes {
        let g = &self.config;
        BumpSizes::initial(
            &g.layout,
            self.max_open_clips,
            g.target_width,
            g.target_height,
            self.max_bytes,
        )
    }
}

/// The reports kept, newest last, at most `capacity`; none at capacity 0.
#[derive(Debug, Default)]
pub(crate) struct Recorder {
    capacity: usize,
    reports: VecDeque<BumpReport>,
}

impl Recorder {
    /// Keep at most `capacity` reports from now on; 0 stops and drops them.
    pub(crate) fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity;
        if capacity == 0 {
            self.reports = VecDeque::new();
        } else {
            while self.reports.len() > capacity {
                self.reports.pop_front();
            }
        }
    }

    /// Keep a report of a frame whose counters came back, if recording; the
    /// oldest goes when full.
    pub(crate) fn record(
        &mut self,
        counters: &BumpAllocators,
        config: &ConfigUniform,
        clips: u32,
        max_bytes: u64,
    ) {
        if self.capacity == 0 {
            return;
        }
        if self.reports.len() == self.capacity {
            self.reports.pop_front();
        }
        self.reports.push_back(BumpReport {
            counters: *counters,
            given: BumpSizes::of(config),
            config: *config,
            max_open_clips: clips,
            max_bytes,
        });
    }

    /// The reports kept, oldest first; recording goes on.
    pub(crate) fn take(&mut self) -> Vec<BumpReport> {
        self.reports.drain(..).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counters(lines: u32) -> BumpAllocators {
        BumpAllocators {
            lines,
            ..BumpAllocators::default()
        }
    }

    fn lines(reports: &[BumpReport]) -> Vec<u32> {
        reports.iter().map(|r| r.counters.lines).collect()
    }

    #[test]
    fn nothing_is_kept_until_asked() {
        let mut recorder = Recorder::default();
        recorder.record(&counters(1), &ConfigUniform::default(), 0, u64::MAX);
        assert!(recorder.take().is_empty());
    }

    #[test]
    fn the_newest_are_kept_up_to_the_capacity_and_a_take_drains() {
        let mut recorder = Recorder::default();
        recorder.set_capacity(2);
        let config = ConfigUniform {
            target_width: 64,
            lines_size: 100,
            ..ConfigUniform::default()
        };
        for n in 1..=3 {
            recorder.record(&counters(n), &config, 5, u64::MAX);
        }
        let reports = recorder.take();
        assert_eq!(lines(&reports), [2, 3]);
        assert_eq!(reports[0].given, BumpSizes::of(&config));
        assert_eq!(reports[0].config.target_width, 64);
        assert_eq!(reports[0].max_open_clips, 5);
        assert_eq!(reports[0].needed().lines, 2);
        assert_eq!(
            reports[0].initial(),
            BumpSizes::initial(&config.layout, 5, 64, 0, u64::MAX),
            "the rules' answer for the frame's target, scene and nesting"
        );
        assert!(recorder.take().is_empty(), "drained");
        recorder.record(&counters(4), &config, 0, u64::MAX);
        assert_eq!(lines(&recorder.take()), [4], "and still recording");
    }

    #[test]
    fn a_smaller_capacity_keeps_the_newest_and_zero_stops() {
        let mut recorder = Recorder::default();
        recorder.set_capacity(3);
        for n in 1..=3 {
            recorder.record(&counters(n), &ConfigUniform::default(), 0, u64::MAX);
        }
        recorder.set_capacity(1);
        assert_eq!(lines(&recorder.take()), [3]);
        recorder.record(&counters(4), &ConfigUniform::default(), 0, u64::MAX);
        recorder.set_capacity(0);
        recorder.record(&counters(5), &ConfigUniform::default(), 0, u64::MAX);
        assert!(recorder.take().is_empty());
    }
}
