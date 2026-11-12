// Copyright 2026 the Vello Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! The build-time naga IR (`naga_ir`, rinch #2468) is the module of each shader's own WGSL.

#![cfg(feature = "naga_ir")]

use vello_shaders::SHADERS;

/// Every shader carries IR that decodes with the naga linked here, and re-encodes to exactly the
/// bytes of a fresh parse of its WGSL: the module wgpu gets is the one the WGSL would have given.
#[test]
fn every_shader_carries_the_module_of_its_wgsl() {
    let all = SHADERS.all();
    assert!(all.len() > 15, "only {} shaders", all.len());
    for shader in all {
        let name = &shader.name;
        assert!(!shader.naga_ir.is_empty(), "{name}: no naga IR");
        let decoded: naga::Module = postcard::from_bytes(&shader.naga_ir)
            .unwrap_or_else(|e| panic!("{name}: IR does not decode: {e}"));
        let parsed = naga::front::wgsl::parse_str(&shader.wgsl.code)
            .unwrap_or_else(|e| panic!("{name}: WGSL does not parse: {e}"));
        assert_eq!(
            postcard::to_allocvec(&decoded).unwrap(),
            postcard::to_allocvec(&parsed).unwrap(),
            "{name}: the IR is not the module of its WGSL"
        );
        assert_eq!(
            &*postcard::to_allocvec(&parsed).unwrap(),
            &*shader.naga_ir,
            "{name}"
        );
    }
}
