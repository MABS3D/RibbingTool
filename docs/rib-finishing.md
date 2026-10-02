# Rib finishing and stable pattern placement

The graph engine now finishes the connected network before rounding its top
edges. Previously, blending already capped ribs lifted their crowns; clipping
that bulge to the target height introduced a sharp edge through the top fillet.
Root blends also raised tapered endpoints that the mapping preview showed at
zero height.

Wall and crown distances share one ribbon query. Junction and root blends act
on locally extended walls, then an inward circular fillet joins the network to
its crown envelope. The envelope uses compact lateral support and a normalized
soft minimum: equal-height crowns remain at their prescribed height, nearby
guide directions meet smoothly, and a distant tall rib cannot lift a short tip.
The root radius decreases continuously where the remaining height cannot
accommodate the requested fillet. No extra user parameter is required.

The selection frame now integrates first and second moments over triangle
areas. Refining a corner of an unchanged planar sheet previously displaced its
origin by 14.06 mm and rotated it by 1.83 degrees in the regression fixture.
The corrected frame preserves both origin and orientation to numerical
precision. Surface mapping and local-control solvers otherwise retain their
existing algorithms.

## Reproducible checks

```powershell
.venv\Scripts\python.exe -m pytest tests/test_graph_finishing.py tests/test_surface_frame.py
```

The finishing tests measure cross sections and actual extracted material:

- Taper heights with three root radii, including a varying local height field.
- A disappearing root footprint that retains its full fillet away from the tip.
- Tangency at an X-junction crown and continuity between different guide slopes.
- A higher intersecting rib and a remote tall rib that must not affect a run-out.
- A closed rib/body union with one material component and no raised endpoint.

All 13 finishing tests pass. Against the preceding implementation, 12 fail:
the extracted endpoint stands 0.331 mm above the body, and a 2 mm root radius
raises the unsampled tip by 0.562 mm. A separate regression detects the hard
handoff between different crown slopes. Frame checks also cover rigid
placement, far-from-origin coordinates, winding and disconnected selections.

The dashboard checks use part1 face 97 and part2 face 208 with identical
parameters before and after. Both pass Apply, finer reconstruction, body
union and actual float32 STL validation: one final solid and zero collapsed
STL triangles. Some short, attached terminal segments remain on part2.

The assembly reference was also rebuilt with its original five-face selection
and manual rotation control. Apply produced 7,762,744 triangles in 368 seconds
with zero open edges, nonmanifold edges or collapsed triangles. Peak private
memory was approximately 6.1 GiB. This larger check covers Apply only; its
fine export was not run. Concurrent test jobs were active, so these times
are context rather than a controlled performance comparison.
See the [machine-readable measurements](benchmarks/rib-finishing.json).

## Compatibility and remaining limits

Existing recipe dimensions remain supported. Regenerating a recipe made with
the old frame can change its lattice phase and angle; inspect its mapping
preview and adjust orientation/offsets when preserving the old layout matters.

The result is still a sampled mesh. These changes do not produce analytic CAD
fillets or prove manufacturability. Preview and finer export retain different
sampling tolerances. Small, attached terminal segments remain legitimate
geometry and are not silently removed.

Boundary clearance still uses three-dimensional distance to the selected
surface boundary. On a tightly folded return, a geometrically nearby edge can
be far away along the surface and can shorten an internal rib. Intrinsic
boundary distance remains a separate limitation; increasing the flattening
iteration count does not solve it.

The [nTop rib design guide](https://support.ntop.com/hc/en-us/articles/35117560848275-Guide-to-Rib-Design)
informs the separation of graph mapping, extrusion, blending and height
control. There is no matched nTop model for a numerical equivalence claim.
