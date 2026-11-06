---
name: AMR Research Agent
description: A lab notebook for antimicrobial-resistance questions, where every answer is a cited, verified entry.
colors:
  notebook-green: "#173a2c"
  notebook-green-deep: "#0f2b20"
  cover-ink: "#e8f0eb"
  cover-muted: "#a9c4b6"
  cover-line: "#ffffff3d"
  journal-blue: "#1f5fa8"
  ink: "#15212b"
  muted-slate: "#52616a"
  placeholder: "#5f6d74"
  ground: "#f1f5f2"
  paper: "#ffffff"
  rule: "#d3dcd6"
  selection: "#cfe3d7"
  verified-green: "#17643f"
  verified-wash: "#e1f1e7"
  invalid-red: "#a3241a"
  invalid-wash: "#fbe7e4"
  caution-umber: "#7a4b00"
  caution-wash: "#fbf0d9"
  app-blue: "#2f6db5"
  app-wash: "#e4edf8"
  agar-amber-ink: "#9a5f0c"
  agar-amber: "#c7861f"
  agar-wash: "#f7ebd6"
typography:
  display:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Times New Roman, serif"
    fontSize: "1.6rem"
    fontWeight: 600
    lineHeight: 1.25
  headline:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Times New Roman, serif"
    fontSize: "1.4rem"
    fontWeight: 700
    lineHeight: 1.2
  title:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Times New Roman, serif"
    fontSize: "1.15rem"
    fontWeight: 600
    lineHeight: 1.3
  answer:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Times New Roman, serif"
    fontSize: "1.1rem"
    fontWeight: 400
    lineHeight: 1.62
    fontFeature: "\"onum\", \"pnum\""
  lead:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Times New Roman, serif"
    fontSize: "1.15rem"
    fontWeight: 400
    lineHeight: 1.45
  reference:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Times New Roman, serif"
    fontSize: "0.98rem"
    fontWeight: 400
    lineHeight: 1.45
  figure:
    fontFamily: "STIX Two Text, STIX Two, Cambria, Times New Roman, serif"
    fontSize: "1.12rem"
    fontWeight: 600
    lineHeight: 1.3
    fontFeature: "\"lnum\", \"tnum\""
  body:
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica Neue, Arial, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica Neue, Arial, sans-serif"
    fontSize: "0.8rem"
    fontWeight: 600
    lineHeight: 1.3
  tool-call:
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"
    fontSize: "0.8rem"
    fontWeight: 400
    lineHeight: 1.4
rounded:
  hairline: "2px"
  soft: "3px"
  control: "4px"
  segmented: "6px"
spacing:
  xxs: "4px"
  xs: "8px"
  sm: "12px"
  md: "18px"
  lg: "24px"
  xl: "28px"
  xxl: "40px"
components:
  button-primary:
    backgroundColor: "{colors.notebook-green}"
    textColor: "{colors.cover-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.control}"
    padding: "0 20px"
    height: "44px"
  button-primary-hover:
    backgroundColor: "{colors.notebook-green-deep}"
    textColor: "{colors.cover-ink}"
  mode-tabs:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.muted-slate}"
    rounded: "{rounded.segmented}"
    padding: "3px"
  mode-tab-selected:
    backgroundColor: "{colors.notebook-green}"
    textColor: "{colors.cover-ink}"
    rounded: "{rounded.control}"
    height: "44px"
  question-field:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.control}"
    padding: "10px 12px"
  notebook-entry:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.hairline}"
    padding: "26px 34px 32px"
  verification-ok:
    backgroundColor: "{colors.verified-wash}"
    textColor: "{colors.verified-green}"
    rounded: "{rounded.soft}"
    padding: "10px 14px"
  verification-bad:
    backgroundColor: "{colors.invalid-wash}"
    textColor: "{colors.invalid-red}"
    rounded: "{rounded.soft}"
    padding: "10px 14px"
  status-limit:
    backgroundColor: "{colors.caution-wash}"
    textColor: "{colors.caution-umber}"
    rounded: "{rounded.soft}"
    padding: "10px 12px"
  call-app:
    backgroundColor: "{colors.app-wash}"
    textColor: "{colors.ink}"
    typography: "{typography.tool-call}"
    rounded: "{rounded.soft}"
    padding: "7px 10px 8px"
  call-mcp:
    backgroundColor: "{colors.agar-wash}"
    textColor: "{colors.ink}"
    typography: "{typography.tool-call}"
    rounded: "{rounded.soft}"
    padding: "7px 10px 8px"
---

# Design System: AMR Research Agent

## Overview

**Creative North Star: "The Lab Notebook"**

The page is a scientific notebook. A deep green cover strip carries the title, the author and the project links. Below it, cool white paper sits on a faint green-grey ground. Each answer is written up as a notebook entry: the question is the title, the answer is set like a journal article, then come a citation check, Methods (the agent's trajectory drawn as a figure) and numbered References. The serif belongs to the research being reported. The system sans belongs to the controls that run it. Monospace is used only where a tool was actually called.

Density is calm and editorial. One raised sheet of paper holds the result; everything around it is flat, divided by hairline rules rather than boxed. The colour is functional: green means cover and verified, blue means a link or something that ran inside the app, amber means something that crossed into `amr-mcp`, and red marks a claim that can't be trusted. Nothing is decorative. Every tint stands for a fact about the run.

The world rejects the chat-box idiom: there are no message bubbles, no pill chips and no heavily rounded cards. The paper is never cream.

**Key Characteristics:**
- Notebook-green cover strip; cool white paper on a green-grey ground (never cream).
- STIX Two Text for the question, the answer, the references and the figures; system sans for controls; mono only for tool calls.
- Genes and species italicised automatically, as a journal would set them.
- Methods drawn as two lanes split by a dashed MCP boundary; bar length is latency.
- Small radii (2–6px), hairline rules, a single raised sheet.
- Full dark scheme through `prefers-color-scheme`, same roles, retuned values.

## Colors

A restrained scientific palette: one cover green, journal blue for links, and two lane colours (app blue, Mueller-Hinton agar amber) that encode where a tool ran.

### Primary
- **Notebook Green** (notebook-green): the cover strip, the primary button and the selected mode tab. It is the page's identity and its one action colour. Hover deepens to **Notebook Green Deep** (notebook-green-deep). Text on it is **Cover Ink** (cover-ink), secondary text **Cover Muted** (cover-muted), and link underlines on the cover use **Cover Line** (cover-line, translucent white).
- **Journal Blue** (journal-blue): every link, inline citation numbers, the focus ring and the text caret.

### Secondary
- **App Blue** (app-blue, wash app-wash): tools that ran inside this app (`search_literature`). The lane swatch and the latency bar use the full colour; the call block sits on the wash.

### Tertiary
- **Mueller-Hinton Amber** (agar-amber, wash agar-wash, text agar-amber-ink): tools that ran in `amr-mcp`, across the MCP boundary. The full amber draws the dashed boundary, the lane swatch and the latency bars; the darker ink is used for amber text ("via amr-mcp").

### State
- **Verified Green** on **Verified Wash** (verified-green / verified-wash): the citation check when every PMID was returned by a search or tool, and informational status.
- **Invalid Red** on **Invalid Wash** (invalid-red / invalid-wash): unverified citations (inline numbers turn red), errors, and a hairline inset ring round a failed tool call.
- **Caution Umber** on **Caution Wash** (caution-umber / caution-wash): rate limits, the English-only search note, the "trimmed to fit" flag.

### Neutral
- **Ink** (ink): body text and headings.
- **Muted Slate** (muted-slate): metadata, captions, figure labels, reference markers. **Placeholder** (placeholder) is its slightly lighter sibling, used only in the question field.
- **Ground** (ground): the page background, a faint green-grey that sets off the paper.
- **Paper** (paper): the notebook entry, form fields and the mode tabs.
- **Rule** (rule): every hairline divider and field border.
- **Selection** (selection): text selection and the highlight on a targeted reference.

### Dark scheme
The same roles are retuned under `prefers-color-scheme: dark` (values in `.impeccable/design.json`, `colorMeta.*.dark`). The cover becomes a darker green, the paper a deep green-black, and the lane and state colours become their lighter tints so they read on dark paper. In dark mode the primary button and selected tab use a lifted green (#2f6b4f, hover #377a5b) instead of the cover green, so the action does not disappear into the cover.

### Named Rules
**The Colour Is Evidence Rule.** Blue and amber say where a tool ran; green and red say whether a citation holds. Don't use either pair for decoration or emphasis.

**The Cool Paper Rule.** Paper is pure white (dark: deep green-black) on a green-grey ground. Never warm it to cream or parchment.

## Typography

**Display Font:** STIX Two Text (with STIX Two, Cambria, Times New Roman, serif), loaded from Google Fonts in 400, 600, 700 and italics 400, 600.
**Body Font:** the system sans stack (-apple-system, BlinkMacSystemFont, Segoe UI, Helvetica Neue, Arial).
**Label/Mono Font:** ui-monospace, SFMono-Regular, Menlo, Consolas, used only for tool calls and their raw results.

**Character:** STIX Two is the face scientific publishers use, so the research reads like a paper. The plain system sans keeps the controls clearly part of the instrument, not the record.

### Hierarchy
- **Display** (600, 1.6rem, 1.25; 1.35rem below 900px): the question as the entry's title, balanced, max 44ch.
- **Headline** (700, 1.4rem, 1.2): the project title on the cover strip.
- **Title** (600, 1.15rem, 1.3): entry sections (Methods, References) and, at 1.08rem, the "Eval results" heading.
- **Answer** (400, 1.1rem, 1.62; 1.05rem below 900px): the answer text, max 68ch, old-style proportional figures. Citation numbers are superscript sans at 0.68em, 600.
- **Lead** (400, 1.15rem, 1.45): the one-sentence introduction, max 34ch on desktop.
- **Reference** (400, 0.98rem, 1.45): the numbered reference list, max 72ch; journal names italic, the PMID link and notes in 0.8rem sans.
- **Figure** (600, 1.12rem, 1.3, lining tabular figures): measured numbers in the evidence list; the same face at 0.9rem marks round numbers in Methods.
- **Body** (400, 16px, 1.5): controls, notes and captions, mostly stepped down to 0.84–0.92rem.
- **Label** (600, 0.8rem, 1.3): lane headers, button text (0.95rem), tab names (0.92rem).
- **Tool call** (mono, 0.8rem, 1.4): the call signature, tool name in bold; raw results at 0.76rem.

### Named Rules
**The Two Registers Rule.** Serif for what the research says (question, answer, references, measured figures); sans for what the instrument does (controls, metadata, captions). A control set in the serif, or an answer in the sans, breaks the notebook.

**The Journal Italics Rule.** Gene names and species are italicised in questions, answers and reference titles, and journal names are italic in references, as a journal would set them.

## Layout

A two-column notebook spread on desktop inside a 1280px container (padding 28px 24px 40px): a left column of 280–360px holding the lead, mode tabs, question form, example questions and the measured evidence, and a fluid right column holding the entry. Column gap 40px, row gap 28px. The left column is not sticky: it is taller than a typical viewport.

Below 900px the layout becomes one column (padding 18px 16px 32px, gap 22px) and is reordered: the form first, then the entry, then examples, evidence and footer, so the result sits right under the question. The cover links wrap onto their own row. Below 600px the Methods figure drops to a single lane: lane headers hide, each amr-mcp call gains a "via amr-mcp" line, and the dashed boundary stays as the left edge of those calls.

Spacing follows a loose 4px rhythm (4, 8, 12, 18, 24, 28, 40); hairline-ruled lists use 11px vertical padding to reach a 44px touch height. Prose is capped at 68ch, references at 72ch.

## Elevation & Depth

Flat by default, with one exception: the notebook entry is the only raised surface, a sheet of paper lifted off the ground by a very soft two-layer shadow. Everything else separates with hairline rules, tonal washes and the ground/paper contrast.

### Shadow Vocabulary
- **Paper lift** (`box-shadow: 0 1px 2px #15212b0d, 0 8px 24px -12px #15212b1f`): the notebook entry only.
- **Error ring** (`box-shadow: inset 0 0 0 1px` invalid red): a tool call that returned an error.
- **Target halo** (`box-shadow: 0 0 0 4px` selection): a reference reached from its inline number.

### Named Rules
**The One Sheet Rule.** Only the entry casts a shadow. Panels, calls and notices are flat washes.

## Shapes

Corners are barely softened: 2px for the paper and code blocks, 3px for notices, the citation check and tool calls, 4px for inputs and buttons, 6px for the outer segmented-control frame. Lists and sections are divided by 1px rules; the evidence block opens with a 2px ink rule. In Methods, round rows are separated by solid rules, continuation rows by dotted ones, and the MCP boundary is a 2px dashed amber line. Lane swatches are 9px squares with 2px corners; latency bars are 6px tall with 3px ends.

**The Dashed Boundary Rule.** A dashed amber line means "this crossed into amr-mcp". Use it only for that boundary.

## Components

### Buttons
Quiet and solid, in the cover's colour.
- **Shape:** gently squared (4px).
- **Primary:** notebook green with cover ink text, 600 0.95rem sans, 0 20px, min height 44px, aligned right of the keyboard hint.
- **Hover / Focus:** background deepens over 150ms ease-out; focus is the global 2px journal-blue outline offset 2px. Disabled drops to 0.62 opacity with a not-allowed cursor (progress cursor while busy).
- **Link button:** inline actions styled as underlined journal-blue text; inside status notices they take the notice colour at 600.

### Mode tabs (segmented control)
- **Style:** a two-option radio group in a paper frame (1px rule, 6px corners, 3px inner padding, 3px gap). Each option shows a name (600 0.92rem) over a small description (0.76rem).
- **State:** the selected option fills with notebook green and cover ink; unselected options are muted slate and darken to ink on hover. Keyboard focus outlines the option.

### Inputs / Fields
- **Style:** paper background, 1px rule border, 4px corners; the question box is 104px tall minimum, 10px 12px padding, vertically resizable. The select is 36px tall.
- **Focus:** the journal-blue outline sits flush (offset 0) on the textarea.
- **Notes:** helper notes in 0.85rem muted; the language note in caution umber.

### Status notices
- **Style:** 3px corners, 10px 12px padding, 0.9rem, on a state wash with state-colour text: invalid for errors, caution for limits, verified for information. Hidden when empty.

### Example questions
- **Style:** a borderless list divided by rules above and below each item; full-width sans buttons at 0.92rem with 44px height. Hover turns the text journal blue.

### Notebook entry (signature)
The page's centre: one sheet of paper (2px corners, paper lift, 26px 34px 32px padding; 20px 18px 24px on narrow screens).
- **Meta line:** 0.84rem muted run details (recorded or live, date, mode, model), with the run type in 600 ink.
- **Question title:** display serif, italicised genes and species.
- **Answer:** answer serif with superscript citation numbers linking to References; unverified numbers turn invalid red.
- **Citation check:** a washed band with an inline SVG check or alert icon and a bold count ("2 of 2 citations verified."), then a plain sentence on what that means. Verified green, invalid red, or ground/muted when there are no citations.
- **Sections:** Methods and References as title-serif headings with 32px above; captions open with a bold "Fig. 1."
- **References:** a numbered serif list with bold muted markers, italic journal, year and a sans PMID link; extra retrieved sources sit in a collapsed details list.

### Methods lanes (signature)
The agent's trajectory as a figure: a narrow round-number column and two equal lanes, "In this app" (app blue) and "amr-mcp, over MCP" (agar amber), split by the dashed boundary. Each call sits in the lane where it ran, on that lane's wash, with the call signature in mono and a latency bar in the lane colour whose length is the latency. Flags after the timing ("no matches", "trimmed to fit") take caution or invalid colour. Raw results open in a details block on paper. Totals follow in 0.84rem muted tabular figures.
- **Motion:** the only animation on the page. When the trajectory appears, bars draw from the left (`scaleX(0.04 → 1)`, 700ms `cubic-bezier(0.16, 1, 0.3, 1)`, staggered 70ms per call after 120ms), and only under `prefers-reduced-motion: no-preference`.

### Cover strip (navigation)
- **Style:** notebook green across the full width; the serif title, a muted "by" line with the author's links, and project links pushed right. Links are cover ink without underline, carrying a translucent 1px bottom border that turns solid on hover. Focus outlines switch to cover ink. On narrow screens the links wrap to a full-width row.

### Measured evidence
- **Style:** opens with a 2px ink rule; a two-column definition list (3.4rem figure column) of serif figures beside muted 0.88rem descriptions, max 34ch, then text links and a collapsible "How it works" list divided by a rule.

## Do's and Don'ts

### Do:
- **Do** put new research content (questions, answers, references, measured figures) in STIX Two Text and controls in the system sans.
- **Do** colour any tool call by where it ran: app blue on app wash inside the app, agar amber on agar wash across the MCP boundary.
- **Do** report citation trust in the verification band: verified green when every PMID was returned, invalid red otherwise.
- **Do** keep touch targets at 44px minimum height for buttons, tabs and example questions.
- **Do** cap prose at 68ch and references at 72ch.
- **Do** honour `prefers-reduced-motion`: the latency-bar draw is the only motion and it switches off.

### Don't:
- **Don't** warm the paper to cream or parchment; the paper is cool white on green-grey.
- **Don't** use chat bubbles, pill-shaped chips or heavily rounded cards; corners stay at 2–6px.
- **Don't** use monospace for anything but tool calls and their raw results.
- **Don't** give any surface other than the notebook entry a shadow.
- **Don't** use blue, amber, green or red decoratively; each stands for a fact about the run.
