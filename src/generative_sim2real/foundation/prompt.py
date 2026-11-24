"""One edit instruction shared by the open (Qwen) and closed (Gemini) editors."""

_BODY = (
    "{src} is a synthetic colonoscopy frame. {refs} are reference frames of a real colon phantom. "
    "Re-render {src_lower} so that its surface material, colour, specular highlights and "
    "illumination match the phantom in {refs_lower}; copy only their look, never their content: no "
    "object, hole, fold, vessel or layout from {refs_lower} may appear. Keep the geometry of {src_lower} "
    "pixel-exact: do not crop, zoom, shift or rotate; do not add, remove or move any fold, "
    "lumen opening, edge or shadow boundary; do not add borders, text or vignettes. Change only "
    "appearance, never shape. Output a single image with the same framing as {src_lower}."
)
PROMPT = _BODY.format(src='Image 1', refs='Images 2 and 3', src_lower='image 1', refs_lower='images 2 and 3')
# Variant for editors that order the inputs [reference, reference, source frame].
PROMPT_SOURCE_LAST = _BODY.format(src='Image 3', refs='Images 1 and 2', src_lower='image 3',
                                  refs_lower='images 1 and 2')
NEGATIVE_PROMPT = "crop, zoom, border, frame, text, watermark, added folds, removed folds, distorted geometry"
# Depth-conditioned variant: image 1 is the depth control image (near = bright) instead of the synthetic frame.
PROMPT_DEPTH = (
    "Image 1 is a depth map of a colonoscopy view: brighter means nearer to the camera, black means far away "
    "(the dark lumen opening) or no data. Images 2 and 3 are reference frames of a real colon phantom. Generate a "
    "realistic colonoscopy image of exactly the scene in image 1: every fold, lumen opening, edge and surface must "
    "be at the same pixel position and with the same shape as in the depth map. Give it the surface material, "
    "colour, specular highlights and illumination of the phantom in images 2 and 3; copy only their look, never "
    "their content: no object, hole, fold, vessel or layout from images 2 and 3 may appear. Do not crop, zoom, "
    "shift or rotate; do not add borders, text or vignettes. Output a single image with the same framing as image 1."
)
