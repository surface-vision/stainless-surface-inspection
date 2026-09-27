# Round 2 video script — 148 seconds

Use this only if you submit the **video** instead of the deck. The brief allows
120–150 sec video **or** a 6–8 slide PPT, not both.

Voiceover is 344 words, about 2:22 at 145 words per minute. On-screen visuals come
from `out/JSW_Round2_Surface_Defect_Detection.pptx` plus a screen recording of the
live demo. Timings total 2:28, which leaves 2 seconds under the 150 s limit.

| Time | Section | On screen | Voiceover |
|---|---|---|---|
| 0:00–0:12 | Hook | Slide 1, then zoom to the QR code | "A stainless coil is graded at the end of the line. By then, a sliver born at the caster has already paid for hot rolling, annealing, pickling and cold rolling. We built AI that catches it where it is born." |
| 0:12–0:38 | Problem | Slide 2: the value-chain ribbon, then the nickel tile | "A strip at 300 metres a minute is too fast for any inspector to see the whole surface. On stainless that costs more than on carbon steel: every tonne of 304 carries over a lakh of rupees of nickel. So Jindal's question is not only whether there is a defect. It is how to find it early and know which process caused it." |
| 0:38–1:06 | Insights | Slide 3: the defect gallery and owners. Slide 4: the false-alarm bars | "Two insights shaped our design. First, on stainless the surface is the product, and the defect that matters depends on grade and finish, so we map every defect class to its process owner. Second, our own tests showed that a model trained only on lab images wrongly flagged half of all clean mill strip. Training it on real clean strip cut that to under five percent, and it caught nearly twice as many real defects." |
| 1:06–1:36 | Solution | Screen recording: upload a strip image to surface-vision.github.io, show the boxes. Then slide 5: the architecture ribbon | "This is the detector running live in a browser: box, defect type, calibrated confidence and severity. On the line it becomes six stages: capture, a check that the image is steel, detect, score, decide at the coil, then act and learn. Every alarm reaches its owner as a work order, and every operator override retrains the model in under an hour. We buy proven cameras and own the model." |
| 1:36–1:59 | Implementation | Slide 7: the Gantt, with the month-7 line highlighted | "The rollout is gated. It is instrumented on one 300-series cold-rolled line, fine-tuned on Jindal's own strip, and running in shadow mode by month seven. Nothing touches the mill until operators trust it. Estimated cost is 2.5 to 4.2 crore per line, against 1 to 3 million dollars for a commercial system." |
| 1:59–2:22 | Impact and ask | Slide 8: the loss-pool table, then the payback bars, then the ask box | "Even at a conservative 1% downgrade rate, one line pays back in about two years. Across the fleet it protects around 13 crore a year, with full-surface coverage, fewer customer claims and less remelting. Our ask: one line, seven months, and your downgrade data. Let's catch every defect where it is born." |
| 2:22–2:28 | End card | Slide 1, with team name and the QR code | (Silence, or music only.) Team name, members, institution, and the demo URL: surface-vision.github.io |

If a take runs long, drop "We buy proven cameras and own the model" first, then
"and accuracy went up, not down."

## Recording notes

- Record the demo segment at 1920×1080, browser zoomed to 125% so the boxes are
  readable. Use a sample strip image from the demo's built-in set, so the take is
  repeatable.
- Round the numbers in the voiceover — "half" and "under five percent", not 51.7% and 4.7%.
  Keep the exact figures on screen.
- Burn in subtitles. Judges often watch without sound.
- Slide 6 (proof: per-defect accuracy and weak spots) has no spoken segment. If there is
  time left after a take, show it for 3-4 seconds under the Solution voiceover.
- The risk table and KPIs (slide 7, lower half) have no spoken segment. They are deck-only
  depth. If a judge asks "what could go wrong?", that is the answer.
