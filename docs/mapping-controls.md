# Mapping controls before generation

The Mapping studio previews the actual attachment paths and height guides on
the selected STEP faces, before sampling any solid. Local controls prescribe
design intent; they do not represent computed stresses or predict stiffness.

## Workflow

1. Load a STEP file, select the faces and choose **Surface** with engine **auto**.
2. Click **Preview mapping**. Cyan curves show rib attachment paths; amber
   curves show their height, including edge taper. Thickness and fillets are
   constructed by **Apply ribs**.
3. Click **Add control**, then click a selected face. Dragging still orbits the
   camera. Press Escape to cancel placement.
4. Click a numbered point on the model to open its nearby control menu. Edit
   the influence radius, local rotation, spacing multiplier and height
   multiplier there. The menu follows the point while you orbit the camera;
   the sidebar list provides another way to select a point.
5. Drag the graduated rotation ring to change the angle, or type an exact
   value in the menu. Hold **Shift** for 15° steps. **Esc** cancels a drag;
   when not dragging it closes the menu. The angle updates during the gesture,
   then the mapping is recalculated once on release. **Undo edit** reverses
   the whole drag in one step. Controls can also be repositioned, disabled,
   or removed from the menu.
6. Once the preview is valid, **Apply ribs** builds that layout. The same
   controls are stored in the generation recipe and used by fine export.

The rotation ring lies in the tangent plane at the control. Previously traced
paths stay visible as a reference during a drag; the handle and numeric angle
update immediately, and the paths update after release. The menu stays inside
the viewport and closes with its close button or Escape. Visible point labels
remain selectable among existing ribs; hidden labels cannot be picked through
the body.

The controls fade smoothly toward the unchanged base layout. Rotation is an
offset from the existing local pattern direction; spacing and height are
multipliers of the global parameters. Overlapping controls blend. A spacing
multiplier describes the target near the anchor, not a guarantee of perfectly
uniform physical cells on a doubly curved surface.

The preview uses the same clipping and taper code as solid generation. It is
not a solid, does not add an undo step, and does not replace previously built
ribs. Editing after Apply changes the next operation; use the main **Undo**
button before applying again if you want to replace the preceding operation.

## Saving a layout

**Save mapping** downloads a versioned JSON file containing the face selection,
global parameters and all controls, including disabled ones. **Open mapping**
requires the original STEP file: a SHA-256 fingerprint prevents assigning face
IDs from an unrelated model. Opening a layout uses the auto engine.

A local browser draft is also saved for that exact STEP file. Refreshing the
page reconnects to the model held by the local server and restores its draft.
The JSON file is the portable copy; browser storage can be cleared or disabled.

**Close model** in the Model panel unloads the current file and clears the
viewport, face selection, preview, controls and model undo history. Refreshing
the page keeps the workspace empty. The browser's saved mapping draft remains
available when you load the same STEP again.

## Scope of this first increment

- Up to 32 local controls. Regular rectangular, quadmesh, triangular, isogrid
  and hexagonal patterns support them. Stochastic patterns do not yet support
  local controls. Project supports preview but has no local controls.
- Influence follows the selected surface, so an adjacent wall across a narrow
  gap does not receive an edit merely because it is spatially close.
- Existing parameterization seams remain boundaries. This increment does not
  close periodic seams or solve the global mapping problem on every topology.
- The existing border offset remains in parameter space, and path clearance
  uses the existing physical boundary distance. Exact intrinsic offsets around
  curved boundaries are a separate improvement.
- Extreme or incompatible edits are refused before solid construction if the
  map folds or becomes excessively compressed. Increase the affected region,
  reduce the edit or reposition overlapping controls rather than generating
  an invalid result.
- Small radii require more local mesh detail. A mesh budget rejects excessive
  refinement explicitly. Complex selections take longer than simple plates;
  the preview reports its measured calculation time.
- Prepared control supports are cached. Height edits on an already active
  control reuse that support; changes in direction or spacing may require
  additional refinement and a new distance calculation. Large dashboard
  selections can therefore take tens of seconds even after the first preview.
  Each new layout starts from the same canonical support: acceptance and
  geometry do not depend on which angle was tried earlier. Returning to a
  cached layout reuses its validated intrinsic distances.

## API and extension point

`POST /api/mapping/preview` accepts the same `face_ids`, `params` and `engine`
as `/api/ribs`, plus the optional `model_token` returned when loading a model.
It returns world-coordinate `paths[].points` and `paths[].top` as flat XYZ
arrays, projected controls, statistics and warnings. `/api/model` reconnects
the browser to the currently loaded model. Model operations are serialized;
a concurrent operation receives HTTP 409 and leaves the current model intact.

```json
{
  "face_ids": [6],
  "engine": "auto",
  "params": {
    "mapping": "surface",
    "pattern": "isogrid",
    "spacing": 12,
    "height": 4,
    "mapping_controls": [{
      "id": "control-1",
      "face_id": 6,
      "position": [30, 20, 8],
      "radius": 36,
      "angle_deg": 25,
      "spacing_scale": 0.8,
      "height_scale": 1.6
    }]
  }
}
```

Anchors are projected onto their named selected face; points farther than 1 mm
from it are refused. This handles display tessellation differences without
silently attaching a control to another nearby sheet.

A future variable-radius brush can feed manually authored scalar fields and
tangent directions into this pipeline. Painting an intensity alone does not
specify a rib direction: directional strokes or a separate direction channel
will be needed. The brush UI and stress simulation are not implemented here.
