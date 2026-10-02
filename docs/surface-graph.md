# Surface graph ribs

`auto` with **Surface** (the UI default) or **Project** mapping selects the
`graph` construction. Surface follows distances along the selected faces;
Project deliberately preserves the frontal lattice and stretches on steep
walls. The API also retains legacy `unfold` for existing callers. The old
closest-surface pattern field remains available as `implicit` for comparison.

The graph engine intersects finite pattern segments with the selected surface,
connects the resulting curves across triangle and CAD face boundaries, then
extrudes those curves into ribbon surfaces. Three-dimensional distance to the
ribbons controls wall thickness. It does not transfer a 2D pattern between
different closest faces while evaluating the rib volume.

Surface coordinates start from LSCM/ARAP. Small inverted seed triangles are
repaired locally, then 30 SLIM symmetric-Dirichlet iterations balance stretch
with a positive-Jacobian barrier. A failed repair raises an error rather than
building from a folded map. The final map is area-scaled to millimetres and
rigidly aligned to the selection frame using area weights. Disconnected selected
regions receive separate maps. Geometry/frame-keyed caching preserves the map
between preview and export. See the [libigl tutorial](https://libigl.github.io/tutorial/)
and [SLIM project](https://igl.ethz.ch/projects/slim/).

Positive local Jacobians do not imply a globally non-overlapping UV boundary.
The curve construction pulls the pattern back through each original triangle,
so globally overlapping parts of the chart do not switch sheets or get welded
in UV space. A doubly curved surface cannot generally carry perfectly regular
planar cells: some residual stretch is unavoidable with one continuous chart.

The embedded root is constrained by both the body and the selected surface.
Material outside the body is permitted only on the positive side of the selected
faces, so a large root blend cannot print ribs through a thin wall's back face.
Outside material must also be nearer the selected surface than the excluded
faces: root blends cannot spill onto an adjacent unselected face. The embedded
overlap inside the body remains available for joining. Selection boundaries
are never expanded to close unselected strips or holes.
Distance to all ribbons provides a conservative query band; skipping empty space
keeps complete marching cells around the surface.

After extraction, each material component is checked against the same body
tessellation used by export. Components with no stable solid attachment are
removed with an explicit report; a wholly unsupported result is an error.
Small attached ribs are retained. Overlapping components are boolean-unioned
before export, with internal cavities preserved.

Controls:

- **Thickness** is measured across the ribbon wall in 3D.
- **Height** is extrusion length along the guide direction. Junction additions
  are also limited by the body's height offset.
- **Guide smoothing** diffuses extrusion directions over the selected surface
  using area and cotangent weights. It does not change attachment points.
  Zero uses the original surface normals. The default is 3 mm.
- **Root fillet**, **top fillet**, and **junction fillet** are independent.
  Top radius is limited to half the wall thickness and the rib height.
  The network is blended before the top fillet is applied. Nearby guide
  crowns transition smoothly at curved intersections without lifting a flat
  crown. Root radius decreases smoothly as a run-out approaches zero height.
- **Margin** and **taper** use distance to the selected surface's actual open
  boundary. Centerline clearance includes half the wall thickness; root blends
  can broaden the footprint beyond it. Internal endpoints of hexagonal or
  stochastic edges do not taper.

The result remains a sampled mesh. Preview pitch is at most 0.25 mm and at most
one sixth of wall thickness, subject to the 0.08 mm lower limit. Export rebuilds
at three times the preview quality before joining the ribs to the CAD body.
Export reports rebuild errors instead of silently substituting preview meshes.
Graph STEP export contains faceted surfaces, not analytic CAD fillets.
Complete dashboard selections produce millions of triangles and can take
minutes to build; the finer export is substantially more expensive than preview.

The explicit graph improves geometric continuity but is not a general substitute
for structural optimization, manufacturing checks, or a custom guide surface.
Closed or unsupported topology can require a split into smaller selections.
The tests cover finite pattern mapping, metric spacing across a right-angle
fold, cylinders extending beyond the silhouette, positive map orientation,
dimension controls, excluded adjacent faces, and solid union; real dashboard
references are also checked. Boundary ribs are inset in pattern coordinates;
on curved surfaces their offset inherits the map's residual metric distortion.

The selection frame uses integrated surface-area moments, so merely refining
the CAD tessellation does not move or rotate the pattern. This replaces the
older vertex-density-dependent frame: regenerating older recipes can change
their lattice phase and orientation. Review the mapping preview after updating.
See [finishing validation](rib-finishing.md) for the geometric checks and limits.
