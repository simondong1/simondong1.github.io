---
name: writing-simon-tech-blog
description: Creates, revises, validates, and publishes technical posts for Simon Dong's GitHub Pages blog. Use whenever writing a blog post, changing simondong1.github.io, turning a learning conversation into an article, or publishing an LLM architecture tutorial.
---

# Writing Simon's Technical Blog

## First action

Read [BLOG_STYLE.md](BLOG_STYLE.md), inspect the current post and related
posts, then fetch the latest remote branch. Preserve concurrent work.

## Canonical project

- This repository is the canonical source.
- Live site: `https://simondong1.github.io/`
- Draft copies outside this repository are not published posts.

## Article workflow

1. Recover the reader's actual learning sequence and prior feedback.
2. Verify technical claims against primary papers, model configs, source code,
   and measured data.
3. Build the explanation in prerequisite order: placement, motivation,
   smallest useful example, general rule, real dimensions, implementation,
   limitations, and final mental model. Define every hardware, kernel, and
   matrix term before using its acronym. First-time readers must finish the
   lede without already knowing MMA, TMEM, TMA, warpgroup, or SM.
4. Match the existing editorial design. Prefer focused diagrams, MathJax, and
   small interactive visuals over prose walls, profiler dumps, or pseudocode.
5. Update the canonical HTML plus `index.html`, `feed.xml`, `sitemap.xml`, and
   `llms.txt` whenever titles, URLs, dates, or descriptions change.
6. Validate markup, source links, desktop and mobile layout, equations, and
   every interactive control.

Before a worked example, tell the reader what is being computed, what the
example teaches, and whether it is a baseline or the final mechanism.

When the post depends on several GPU or kernel names at once, add a compact
terms card after the lede. Expand each acronym on first use in the body too.

## Performance presentation

- Lead with measured points. Use plain chart titles such as “Prefill latency”;
  put timing details in the caption, not the title.
- Use small, uncluttered dots by default. Put exact values in hover text and
  the nearby result prose; do not print numbers on every dot unless the user
  explicitly requests labels. If points are averages, say so once in the
  caption. For an ordered sweep such as sequence or cache length, connect
  points belonging to the same algorithm when the comparison calls for a
  trend line. Never connect different algorithms or bridge a missing point.
- Present the baseline first, then compare optimized implementations against
  it. Immediately after every evidence image, add short finding-first bullets:
  conclusion first, metric evidence second. A caption does not replace those
  bullets.
- When explaining fusion, show comparable profiler Summary views: the eager
  baseline's many result rows beside the fused implementation's one row.
  A cropped header or Details page does not demonstrate kernel count.
- Keep only metrics that establish the conclusion. Grid size, block coverage,
  elapsed cycles, instruction count, memory path, and occupancy are often
  useful; exhaustive profiler dumps are not.

## Image and overflow rules

- Screenshots and figures must stay inside the article column by default:
  `width: 100%; max-width: 100%; height: auto; display: block`.
- Do not use `100vw`, negative margins, translated full-bleed figures, or a
  width larger than the content column unless the user explicitly requests it
  and both desktop and mobile checks prove that it does not overflow.
- Crop screenshots to the informative application panel. Remove desktop
  backgrounds, black borders, empty panes, and unrelated chrome.
- A screenshot must visibly support its caption. If the claim is “12 kernels
  versus 1,” both row counts must be visible in equivalent Summary views.
- Before publishing, test at a desktop viewport and at 390 px. Assert that
  `document.documentElement.scrollWidth <= window.innerWidth`, every figure's
  bounding box stays inside the content container, all images have non-zero
  `naturalWidth`, and any explicitly requested chart labels do not collide or
  clip.
- If an image overflows or its text is unreadable, fix the layout or recrop the
  source; do not publish and ask the reader to zoom.

## Improve this guide from corrections

- When the user identifies a repeatable writing, chart, screenshot, layout, or
  publishing mistake, fix the page and update this skill or `BLOG_STYLE.md` in
  the same change with a concrete prevention rule.
- Remove or revise older guidance that conflicts with the latest correction;
  do not accumulate contradictory rules.
- Validate the new rule against the page that triggered it before publishing.

## Publishing is required

After every completed blog or site change:

1. Run the relevant validation and visual checks.
2. Fetch the remote branch again and integrate new work safely.
3. Commit all files belonging to the change with a descriptive message.
4. Push to the GitHub Pages branch.
5. Poll the live canonical URL with a cache-busting query until the new content
   is visible, then report the live link.

Do not leave completed work only in a local checkout. Skip publication only
when the user explicitly requests a draft, asks not to publish, or a verified
authentication/permission failure blocks the push.
