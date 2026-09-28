# Vendored browser dependencies

These modules are served locally; the app does not fetch code from a CDN.

| Library | Version | Source | License |
| --- | --- | --- | --- |
| Three.js and addons | r170 | https://github.com/mrdoob/three.js/tree/r170 | three.LICENSE.txt |
| meshoptimizer simplifier (embedded WebAssembly) | 1.3.0 | https://www.npmjs.com/package/meshoptimizer/v/1.3.0 | meshoptimizer.LICENSE.txt |
| three-mesh-bvh | 0.9.15 | https://www.npmjs.com/package/three-mesh-bvh/v/0.9.15 | three-mesh-bvh.LICENSE.txt |

The Three.js imports in addons and three-mesh-bvh point to `./three.module.js`.
The unavailable external source-map reference in three-mesh-bvh is removed.
The meshoptimizer simplifier is distributed unchanged as `meshopt-simplifier.js`.
