# Animal puppet pipeline

How to turn one AI-generated **parts sheet** into a living, tappable 2D puppet: a single self-contained HTML file with moods, blinking, talking mouths and the animal's own signature habits (a frog's throat croak, a porcupine curling into a ball, a bunny munching a carrot).

Four puppets have been built this way. Use them as reference implementations:

| Puppet | Sheet | Parts | Template + build | Habits |
|---|---|---|---|---|
| Glossy frog | `frog2-parts-sheet.webp` | `frog2-parts/` | `puppet_frog.tpl.html` + `build_frog.py` (with `frog2-extra.json`) | throat croak, tongue catches a fly, hop + leg kick + splash, wave |
| Porcupine | `porc-parts-sheet.webp` | `porc-parts/` | `puppet_porc.tpl.html` + `build_porc.py` | quills ripple/bristle, poke → curl into ball, nose sniff, ladybug + eyes follow, wave |
| Bunny | `bunny-parts-sheet.webp` | `bunny-parts/` | `puppet_bunny.tpl.html` + `build_bunny.py` | ears bend/twitch/droop/trail, carrot munch bite by bite, nose wiggle, tail wag, hop |
| Flat frog | `frog-parts-sheet.webp` | `frog-parts/` | `puppet_frog.tpl.html` + `build_frog.py` (defaults) | croak, fly, hop |

Outputs: `*-puppet.html` in this folder. They are design prototypes. **None of them ship in the Mukaan app yet.** The fox was tried in the app and then removed.

---

## 1. The idea in one paragraph

Real 2D rigs (Live2D, Spine, Rive) do two things: they **deform one painting with a mesh**, and they **swap painted pieces** for the parts that change shape (eyes, mouths, poses). We do the same in WebGL:

- The full character is drawn once as a 48×48 textured mesh. The vertex shader bends regions of it: ears, quills, throat, tail, nose, head tilt, breathing.
- The face is erased from that painting, and painted eye and mouth pieces are drawn on top and swapped per mood.
- For a pose the mesh can't reach (a raised paw), one limb is erased from a copy of the painting, and a posed limb piece is drawn in its place.

An early attempt assembled the character from loose sheet pieces. It stopped looking like the character: pieces drift, joints show, and the head becomes a ball on a stick. **Always warp the full master drawing, and swap only what must change shape.**

## 2. Prerequisites

- **Python 3.11:** `numpy`, `pillow`, `opencv-python`, `scipy`, `torch`, `transformers>=5` (for SAM 2.1).
- **LaMa inpainting:** the TorchScript model at `~/.cache/lama/big-lama.pt`, md5 `e3aa4aaa15225a33ec84f9f4bc47e500`. It's the IOPaint release file. Do **not** `pip install simple-lama-inpainting`: it forces pillow<10 and numpy<2. The scripts load the `.pt` directly.
- **SAM 2.1:** `facebook/sam2.1-hiera-large` through `transformers` (`Sam2Model`, `Sam2Processor`). It downloads on first use.
- **A way to view pages:** `file://` is blocked in Playwright and flaky in the in-app browser pane. Serve the folder over HTTP instead. `projectf/design` is usually already served on `http://localhost:3100` (launch config `mukaan-design`), so open `http://localhost:3100/mascot/<name>-puppet`. Otherwise run `python3 -m http.server` from this folder.

## 3. Step 0: generate the parts sheet

Paste this into ChatGPT (or any strong image model) and change only the first two lines. The model decides the anatomy and the signature habits by itself.

```
ANIMAL: frog
COLOUR: (optional — leave blank to use the animal's natural colours)

Create ONE character sheet image for a 2D cut-out animation puppet of a cute baby version of the ANIMAL above (use COLOUR as its main colour if given).

BEFORE DRAWING, WORK OUT FOR THIS ANIMAL:
- Its movable body parts (e.g. ears, tail, arms, legs, wings, fins, tentacles, antlers, spines) — whatever this animal really has.
- 2–3 SIGNATURE ANIMATIONS that are iconic for this animal (e.g. a frog's throat sac puffing and its tongue flicking; an octopus waving one tentacle and blowing bubbles; a cat's tail swish and paw lick; an owl's head swivel and wink; a penguin's flipper flap; a bunny's nose wiggle and ear flop; a pufferfish inflating), and the extra drawn pieces each one needs (e.g. an inflated and a deflated throat sac, a tongue, a raised paw, a puffed-up version of a body part, a small prop like a fly or a leaf).

ART STYLE (identical on every piece): kawaii children's-book illustration. Oversized round head about 60% of total height, small soft body. HUGE glossy round eyes, each about a quarter of the face width: deep black pupil, thick warm amber-gold ring, one large white highlight top-left and one small highlight bottom-right, tiny lashes at the outer corners. Tiny nose, tiny "w"-shaped smile, soft pink blush ovals on the cheeks. Soft warm-brown hand-drawn outline (never black), gentle painterly shading, fine fuzzy grainy texture, muted warm pastel colours. Front view, symmetrical, facing the viewer, lit from the top-left.

SHEET LAYOUT: plain flat warm off-white background (#FBF3E4). No text, labels, numbers, borders, drop shadows or sticker outlines. Every piece separated by generous empty space — nothing touches or overlaps anything else. All pieces at EXACTLY the same scale as the full character. Pieces drawn flat and straight, fully visible, crisp edges.

LEFT THIRD — THE FULL CHARACTER: the complete animal sitting upright, full body, large and centred. Pose designed for animation: limbs, ears, tail, wings etc. held slightly away from the body so no part overlaps another part or the face.

RIGHT TWO-THIRDS — THE PARTS, in this row order:
Row 1 — MAIN PIECES: (a) the full character with NO eyes and NO mouth (blank face: skin/fur, blush and brows only), identical to the full character otherwise; (b) the body alone with every movable part removed and the joins smoothly closed.
Row 2 — MOVABLE PARTS: each movable part this animal has, drawn separately and COMPLETE, including any area normally hidden where it joins the body (finish that end with a soft rounded edge). Left/right pairs drawn as mirror pairs.
Row 3 — EYES, each drawn as a matching pair at the same spacing as on the face: eyes open (normal glossy); eyes half-closed and sleepy; eyes fully closed (soft lids in the skin colour, gentle downward lash line); happy eyes (upward ^ ^ arcs); eyes looking left; eyes looking right.
Row 4 — MOUTHS, same size as the original: small "w" smile; open happy laugh with pink tongue; small round "o"; tiny frown; talking shapes "A" wide open, "E" wide flat, "O" round, "M" closed lips.
Row 5 — SIGNATURE ANIMATION PIECES: every extra piece the signature animations need (for example: deflated and inflated throat sac; tongue out; the animal with one limb raised in a wave; a puffed-up body part; a small prop the animal interacts with). Each state of the same part drawn at the same position and size so it can be swapped in place.

CONSISTENCY: every piece must look cut from the full character — same colours, outline weight, texture, lighting and scale.
```

**What models actually return:**
- Row 1 (a) usually comes back as a blank *head*, not a blank full body. That's fine: we erase the face ourselves.
- The eye and mouth pieces are drawn at about 0.7–0.8× the scale of the full character. Auto-rig finds the real scale.
- The full character's painted mouth is often **not** one of the eight mouth pieces (the frog2 lip line, the porcupine and bunny ‿‿). We keep it as `mouth_orig` for the calm face.
- Closed and happy eyes come with pale peach discs behind them. `clean_face.py` removes them.
- Pieces sometimes touch and merge into one component (the bunny's right eye touched its ground shadow). Split them by hand (step 2).
- If the model crams or skips pieces, ask for two images, both containing the full character: image 1 with rows 1–2, image 2 with rows 3–5.

Save the sheet here as `<animal>-parts-sheet.webp`.

## 4. The pipeline, step by step

Use a short prefix per animal, e.g. `owl`. All commands run from this folder. After every step, **look at the output image**; each step has a known way to go wrong.

### Step 1: cut the sheet into components

```bash
python3 cut_sheet.py owl-parts-sheet.webp owl-parts
```

It measures the background from the sheet edges, turns distance from it (in LAB) into foreground, and labels connected components. It prints one line per component, `[x, y, w, h, label]`, sorted by row then x, and saves `_boxes.json` and `_lbl.npy`.

Map each label to a piece by comparing the printed boxes with the sheet. The rows are in prompt order, and within a row, left to right.

### Step 2: name the pieces

Write `owl-map.json` (see `bunny-map.json` / `porc-map.json`):

```json
{"names": {"3":"full", "1":"head", "16":"eye_open_l", "15":"eye_open_r", "...":"..."}, "soft": []}
```

**Required names** (the templates and scripts expect them):
- `full`
- `eye_{open,half,closed,happy}_{l,r}`
- optional look eyes: `eye_lookA_{l,r}` (left) and `eye_lookB_{l,r}` (right)
- `mouth_{smile,laugh,o_small,frown,open,wide,O,w}`

Map these to sheet order: smile, laugh, "o", frown, then A→`open`, E→`wide`, O→`O`, M→`w`. Name the signature pieces freely (`carrot`, `ball_peek`, `paw_up`, `hand_wave`, `leg_kick`, `fly`, `tongue_a`, `drop_a`…).

```bash
python3 name_parts.py owl-parts-sheet.webp owl-parts owl-map.json
```

This writes one RGBA PNG per piece (8 px padding) and `contact.jpg` in the scratch folder. **Open the contact sheet** and check every name against its picture.

**If two pieces merged into one component,** edit `_lbl.npy` before naming. Reassign the stray pixels to a new label (e.g. 99), restrict it by x/y, and append its box to `_boxes.json`. See the bunny case: the eye below the ground line was split off at y ≥ 628, x 170–275. Then keep only the largest connected blob of that piece's alpha.

### Step 3: clean the face pieces

```bash
python3 clean_face.py owl-parts eye_happy_l eye_happy_r eye_closed_l eye_closed_r
```

- The listed eye pieces become **line only**: just the dark lash and arc lines, with no disc. Always list the happy eyes.
- List the closed eyes too, unless they're drawn as skin-coloured lids that match the face (they almost never do; pasted lids look like stickers).
- All eight mouths lose their pale halo.

### Step 4: auto-rig the face

```bash
PARTS=owl-parts/ python3 autorig.py
```

This does masked LAB template matching of `eye_open_l/r` and `mouth_smile` against `full.png` over a range of scales, and writes `rig.json`.

- **Eyes:** reliable. Check that their scale and position make sense.
- **Mouth:** it fails on most sheets (it lands somewhere random, with a tiny scale). Find the mouth by hand:

```bash
python3 grid.py owl-parts/full.png /tmp/owl-mouth.jpg 10  X0 Y0 X1 Y1 4   # zoomed 10px grid around the face
python3 set_mouth.py owl-parts CX CY SCALE
```

Put the swap mouths slightly **below** the painted mouth line; open mouths hang downward. Use a scale around 0.95–1.2, whatever keeps the big "A" mouth clear of the nose.

### Step 5: erase the face from the full character

```bash
PARTS=owl-parts/ MOUTHBOX=x0,y0,x1,y1,Lmax python3 blank_face.py 16
python3 face_preview.py owl-parts /tmp/owl-face.jpg orig open o_small
```

- `blank_face.py` masks both open eyes (dilated by N px, here 16) and the mouth, then LaMa-inpaints `full_blank.png`.
- **MOUTHBOX** masks only the dark mouth-line pixels (luminance < Lmax) in that box. Use it when the painted mouth isn't a sheet piece, which is almost always. The nose and philtrum survive as long as they're outside the box. Without MOUTHBOX it masks the rig's `mouth_smile` footprint.
- **Eye sockets:** LaMa often paints them slightly pale.
  - The frog2 and porcupine sheets needed `PARTS=… RAD=80 python3 tone_fill.py` afterwards, which pulls the fill's low-frequency colour toward the surrounding skin.
  - The bunny looked **worse** with it (muddy), so it kept plain LaMa.
  - Try both and compare.
  - If you see a bright rim, re-run `blank_face.py` with a bigger dilation (20–22) before trying tone_fill again.
- **`face_preview.py`:** renders the blank with the eye and mouth swaps placed. It's the fastest check that the face lines up.

### Step 6 (optional): the painted mouth, and a limb removed for a pose

For a wave or kick, mask the limb that will be replaced:

```bash
python3 grid.py owl-parts/full.png /tmp/owl-grid.jpg 50          # read coordinates
# owl-limbs.json: {"arm_r": {"box":[x0,y0,x1,y1], "pos":[[x,y],...4], "neg":[[x,y],...4]}}
python3 sam_limb.py owl-parts owl-limbs.json /tmp/owl-limbs.jpg  # check the overlay
PARTS=owl-parts/ BELLY=600 MOUTHCUT=y0,y1,x0,x1 python3 limb_variants.py
```

`limb_variants.py` writes three things:
- **`full_noarm.png` / `full_noleg.png`:** `full_blank` with that limb LaMa-erased.
  - An **arm** in front of the body keeps its whole area opaque above the `BELLY` line. Below it, the ground shadow is regrown with `cv2.inpaint`, not left as a hole.
  - A **leg** at the outer edge becomes transparent beyond the body's own outline.
- **`limb_<name>.png`:** the limb cut out of the original painting.
- **`mouth_orig.png`:** the painted mouth, cut as the difference between `full` and `full_blank` inside `MOUTHCUT`. It prints the centre; put that in the build's `mouthOrig`.

For an animal with no limb pose, still cut `mouth_orig`. `limb_variants.py` needs a `limb-masks.npz`, so the bunny's mouth was cut with the same 6-line diff-and-crop snippet run by hand (full vs full_blank inside the mouth box, alpha `clip((diff-12)/30)`).

### Step 7: decide the signature habits

Every habit is built from four techniques. Pick the cheapest one that reads.

| Technique | How | Used for |
|---|---|---|
| **Warp a region** of the full mesh | a vertex-shader displacement weighted by a smooth mask (Gaussian blob, ring, split band) | throat sac bulge, quill bristle and ripple, ear bend, nose wiggle, tail wag, breathing, head tilt |
| **Swap painted pieces** | draw a different texture in place, or cross-fade two | eyes and mouths per mood, curl into a ball, `full_noarm` + raised paw |
| **Bend a limb piece** | rotate each vertex about the limb's root by `amt·(dist/len)²` | wrist flap on a wave, foot flick on a kick |
| **Code particles and props** | a small piece drawn and moved by JS (path, gravity, fade, clip) | fly, ladybug, splash drops, carrot bites (fragment `clipV` trims the tip) |

Also give the animal:
- **An idle schedule:** random blinks and a habit every few seconds.
- **A tap reaction:** frog catches the fly or hops; porcupine curls; bunny hops.
- **Buttons** for each habit, and six moods (calm, happy, sleepy, surprised, grumpy, talking). Each mood also sets habit parameters: ear pose, quill bristle, speed.

Tie behaviour to the character. The eyes should follow a prop (look-left/right pieces), and a habit should end in an emotion (happy eyes after the gulp, a laugh after uncurling).

### Step 8: write the template and build

Start from the closest existing template and copy its shared engine unchanged:
- WebGL setup, mesh, texture loader
- `draw(name, at, root, rot, sx, sy, {mode, follow, op, bend, clip})`
- moods and blink
- the frame loop

Then replace the vertex-shader warp and the signature section. The build script embeds the textures as webp data URIs and fills in `/*TEX*/`, `/*RIG*/` and `/*VIEW*/`.

**Rules the engine depends on:**
- **One coordinate space:** the full character's own pixels. Every rig value (pivots, centres, radii, bands) is measured on `full.png` with `grid.py`.
- **`mode:1`:** only the full body mesh gets it.
- **`follow:1`:** face pieces get it, so they ride exactly the same warp and tilt and stay glued on. Anything you warp near the face must also run on `follow` items.
- **Split masks:** keep a still gap between two warped regions, e.g. `smoothstep` split bands so the head top between the two ears doesn't tear.
- **Rotation sign:** in `mat2(c,s,-s,c)*v`, a positive angle is clockwise on screen (y points down). A left ear falls outward with a negative angle; a right ear with a positive one.
- **Reduced motion:** honour `prefers-reduced-motion`. Hold the mood poses statically and stop all loops.
- **Viewport:** leave room in `VIEW` for everything that leaves the body (tongue, kick, hop height, props).

### Step 9: verify

Load `http://localhost:3100/mascot/<name>-puppet` (Playwright works well) and check the console for errors.

To inspect an action, **freeze time**: hold its start time in the past from a fast `setInterval`, then screenshot. Top-level `let` state is reachable from `page.evaluate`.

```js
window._h = setInterval(() => { waveT = performance.now()/1000 - 0.39*WAVE_DUR }, 2)   // hold the wave at 39%
puffKeys = [[performance.now()/1000 - 100, 200, 1.3]]                                   // hold the throat puff at peak
```

A hidden browser pane throttles `requestAnimationFrame`, and a frozen-looking screenshot usually means exactly that.

**Checklist:**
- Calm matches the sheet.
- Every mood.
- Each habit mid-motion.
- Nothing cut off by the view.
- No seams or ghost outlines around swapped pieces.
- Tap works.

## 5. Worked commands (real runs)

```bash
# porcupine
python3 cut_sheet.py porc-parts-sheet.webp porc-parts
python3 name_parts.py porc-parts-sheet.webp porc-parts porc-map.json
python3 clean_face.py porc-parts eye_happy_l eye_happy_r eye_closed_l eye_closed_r
PARTS=porc-parts/ python3 autorig.py
python3 set_mouth.py porc-parts 248 360 0.95
PARTS=porc-parts/ MOUTHBOX=218,339,278,360,150 python3 blank_face.py 16
PARTS=porc-parts/ RAD=80 python3 tone_fill.py
python3 sam_limb.py porc-parts porc-limbs.json /tmp/porc-limbs.jpg
PARTS=porc-parts/ BELLY=600 MOUTHCUT=336,364,212,284 python3 limb_variants.py
python3 build_porc.py

# glossy frog (painted lip line, closed lids drawn as lines, arm + leg poses)
PARTS=frog2-parts/ MOUTHBOX=145,212,362,244,185 python3 blank_face.py 22 && PARTS=frog2-parts/ RAD=96 python3 tone_fill.py
python3 sam_limb.py frog2-parts frog2-limbs.json /tmp/frog2-limbs.jpg
PARTS=frog2-parts/ MOUTHCUT=205,250,140,370 python3 limb_variants.py
PARTS=frog2-parts/ OUT=frog2-puppet.html CHIN=295 EXTRA="$(cat frog2-extra.json)" python3 build_frog.py
```

## 6. Lessons that cost time

- **Image-model edits:** they're good at global changes (close the eyes, remove all tentacles) but bad at "remove only these two of six similar parts"; they redraw the whole region. For local removals use **SAM mask + LaMa**.
- **SAM prompts:** box plus several positive and negative points beats single clicks, and `hiera-large` gives cleaner edges than small.
- **Alpha holes:** background-distance alpha makes white highlights transparent. Fill interior holes (`binary_fill_holes`) on anything opaque.
- **Loose pieces:** a character assembled from separate sheet pieces loses its identity. Warp the master.
- **Pale discs:** closed and happy eyes on a pale disc read as stickers. Use line-only pieces over the erased skin.
- **Hard alpha cuts:** a hard alpha edge at a "belly line" looks like a paper cut. Fade it, or regrow the neighbouring shadow.
- **Stiff limbs:** a limb rotated as one stiff piece reads as "stuck". Always lift it in and out from its rest pose, swing it about ±24°, and bend it toward the tip so the wrist or foot lags.
- **Kling credits:** each job is charged. Ask the user before generating.

## 7. Known gaps and next steps

- **Four near-copies:** the four templates share about 70% of their code. Next step: one engine file, plus one recipe per animal (rig numbers, masks, warp terms, habit schedule), so a new animal is a recipe, not a template.
- **Manual steps that remain:**
  - Naming the pieces (`*-map.json`).
  - The mouth position (it failed on all four sheets).
  - Limb prompt points and attach points (`root` / `at` / `rot` / `s` for wave and kick pieces).
  - Picking between `tone_fill` and plain LaMa.
- **App port:** the app would need Skia `drawVertices` (react-native-skia is installed) with the same shader maths, or a pre-rendered sprite atlas. Nothing is in the app today.

---

## 8. Prompt for another agent

Copy this to a fresh agent together with the sheet image:

```
You are building an animated 2D puppet of an animal from a parts-sheet image, using the pipeline in
/Users/pradhumngupta/Desktop/Coding/projectf/design/mascot/PUPPET_PIPELINE.md. Read that file completely first,
then read the closest reference implementation it lists (template + build script + map JSON).

Input sheet: <path to the sheet image>. Animal prefix: <prefix, e.g. owl>.

Do, in order, looking at the output image after every step (never assume a step worked):
1. Copy the sheet to <prefix>-parts-sheet.webp and run cut_sheet.py; map every component to a piece name and
   write <prefix>-map.json; run name_parts.py and check the contact sheet; split merged components if any.
2. clean_face.py (happy + closed eyes line-only), autorig.py, then fix the mouth position with grid.py + set_mouth.py.
3. blank_face.py (with MOUTHBOX for a painted mouth line); compare plain LaMa vs tone_fill.py and keep the better;
   verify with face_preview.py.
4. Cut mouth_orig (limb_variants.py MOUTHCUT, or inline). If the sheet has a raised-limb piece, mask that limb with
   sam_limb.py and make full_no<limb>.png with limb_variants.py.
5. Choose 2–4 signature habits for this animal from the sheet's row 5 using the four techniques (warp region, swap
   pieces, bend limb, code props). Write puppet_<prefix>.tpl.html + build_<prefix>.py from the closest template,
   keeping the shared engine and conventions (full-pixel coordinates, mode/follow, reduced motion, six moods, tap).
6. Build, serve over HTTP (localhost:3100/mascot/… or python3 -m http.server), and verify every mood and every habit
   mid-motion with the freeze-time trick; fix anything that looks stuck, cut off, seamed or detached.

Rules: work only inside projectf/design/mascot; do not touch the Mukaan app repos; do not spend Kling or other paid
generation credits without asking. Report what each habit does, which sheet pieces were unused, and any value you
had to hand-tune.
```
