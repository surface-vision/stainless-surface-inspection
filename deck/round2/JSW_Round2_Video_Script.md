# Stainless Spark Round 2: video script (about 148 seconds)

Voiceover at about 145 words a minute. On-screen visuals come from the Round 2 deck (JSW_Round2_Surface_Defect_Detection.pptx) plus a screen recording of the live demo. Timings add up to 148 s, which leaves 2 s of slack under the 150 s limit.

| Time | Section | On screen | Voiceover |
|---|---|---|---|
| 0:00–0:12 | Hook | Cover slide, then zoom in on the QR code | "A stainless coil is graded at the end of the line. By then, a sliver born at the caster has already paid for hot rolling, annealing, pickling and cold rolling. We built AI that catches it where it is born." |
| 0:12–0:38 | Problem | Slide 2: the value chain, then the nickel number | "A strip at 300 metres a minute is too fast for any inspector to see the whole surface. On stainless that costs more than on carbon steel: every tonne of 304 carries over a lakh of rupees of nickel. So Jindal's question is not only whether there is a defect. It is how to find it early and know which process caused it." |
| 0:38–1:06 | Insights | Slide 3: the grade matrix. Slide 4: the false-alarm bars | "Two insights shaped our design. First, on stainless the surface is the product, and the defect that matters depends on grade and finish, so we map every defect class to its process owner. Second, our own tests showed that a model that has never seen clean steel flagged 94% of clean frames. Training on real clean strip brought that down to 32%, and accuracy went up, not down." |
| 1:06–1:36 | Solution | Screen recording: upload a strip image to surface-vision.github.io and show the boxes. Then slide 5: the architecture ribbon | "This is the detector running live in a browser: box, defect type, calibrated confidence and severity. On the line it becomes six stages: capture, a check that the image is steel, detect, score, decide at the coil, then act and learn. Every alarm reaches its owner as a work order, and every operator override retrains the model in under an hour. We buy proven cameras and own the model." |
| 1:36–1:59 | Implementation | Slide 6: Gantt with the month-7 line highlighted | "The rollout is gated. It is instrumented on one 300-series cold-rolled line, fine-tuned on Jindal's own strip, and running in shadow mode by month seven. Nothing touches the mill until operators trust it. Estimated cost is 2.5 to 4.2 crore per line, against 1 to 3 million dollars for a commercial system." |
| 1:59–2:22 | Impact and ask | Slide 8: the heatmap, then the payback bars, then the ask box | "Even at a conservative 1% downgrade rate, one line pays back in about two years. Across the fleet it protects around 13 crore a year, with full-surface coverage, fewer customer claims and less remelting. Our ask: one line, seven months, and your downgrade data. Let's catch every defect where it is born." |
| 2:22–2:28 | End card | Cover slide, with team name and the QR code | (Silence, or music only.) Team name, members, institution, and the demo URL: surface-vision.github.io |

The voiceover is 344 words, about 2:22 at 145 words a minute. If a take runs long, drop the sentence "We buy proven cameras and own the model" first, then "and accuracy went up, not down."

## Recording tips

- Record the demo segment at 1920×1080, with the browser zoomed to 125% so the boxes are readable. Use a sample strip image from the demo's built-in set, so the take is repeatable.
- In the voiceover, round the numbers: 94% and 32%, not 93.7% and 32.5%. Use the exact figures on screen.
- Burn subtitles into the video, because judges often watch without sound.
