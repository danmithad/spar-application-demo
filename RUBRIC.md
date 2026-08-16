# Ground-truth labeling criterion (hand-written, Claude Fable 5)

Passage-level, multi-label. A passage may receive any subset of
`{adoption, risk, vendor, harm}`, including the empty set. Labels are assigned
from the text of the passage alone — no document context, no outside knowledge
about the company. The word "AI" (or ML, GenAI, LLM, etc.) appearing is never
sufficient by itself for any label.

"AI" here covers: artificial intelligence, machine learning, generative AI,
LLMs, computer vision, predictive analytics explicitly framed as ML/AI, and
automation *when the passage itself frames it as AI-driven*.

## adoption — the company uses or builds AI itself

Assign when the passage asserts that the reporting company **uses, deploys,
develops, pilots, or integrates** AI in its own operations, products, or
services. Present tense or committed near-term ("we are rolling out"), general
("across our business") or specific ("fraud detection models in our payments
stack") both count.

Do **not** assign for:
- Market/industry commentary with no claim about the company's own use
  ("AI is transforming retail").
- Pure aspiration with no commitment ("we may explore AI opportunities").
- Descriptions of *customers'* or *competitors'* use of AI.

Boundary rule: "we are investing in AI" **with any operational object** (tools,
teams, capabilities, products) → adoption; a bare financial allocation with no
operational claim → no label.

## risk — AI as a source of potential adverse outcome for the company

Assign when the passage identifies AI as a **source of risk to the company**:
operational failure, model error/hallucination, bias, security/adversarial use,
regulatory/compliance exposure, reputational damage, IP/copyright exposure,
workforce disruption the company treats as a risk to itself, or competitive
displacement by AI-enabled rivals. Boilerplate risk-factor language counts —
substantive mechanism is *not* required (that distinction is a separate axis in
the Observatory pipeline, not this label).

Do **not** assign for:
- The mere presence of the passage in a risk-factors section; the sentence
  itself must tie AI to a potential adverse outcome.
- AI framed only as a risk *mitigation* tool ("we use ML to detect fraud") —
  that is adoption, not risk.
- Risks to society/others with no link to consequences for the company — unless
  the passage draws that link (e.g. via liability or reputation), in which case
  risk applies.

## vendor — reliance on a third party for AI capability

Assign when the passage states the company **obtains AI capability from, or
depends on, an external party**: named providers (OpenAI, Anthropic, cloud AI
services), unnamed "third-party AI models/tools", AI features embedded in
licensed software the passage highlights, or suppliers of AI-critical compute
(e.g. GPU suppliers) *when the passage frames them as something the company
depends on*.

Do **not** assign for:
- Generic cloud/IT outsourcing with no AI component stated.
- The company *selling* AI to others (that direction is adoption).
- Partnership announcements that are purely commercial go-to-market deals with
  no capability dependency stated.

`vendor` frequently co-occurs with `adoption` (using a third-party model in
your product is both) and with `risk` (third-party dependency listed as a risk
factor) — assign all that apply.

## harm — a realized or concretely materializing AI-related negative event

Assign when the passage reports a negative AI-related event that **has happened
or is concretely in progress**: an incident or outage caused by an AI system,
litigation or regulatory action over the company's AI (filed, not hypothetical),
a discovered bias/safety failure, realized financial loss, restructuring or
layoffs the passage attributes to AI, a data leak via an AI tool.

Distinguish from risk by tense and specificity: **risk = could happen;
harm = happened / is happening** (pending lawsuits count; "we may face
lawsuits" does not). A passage reporting a past incident *and* warning of
future recurrence gets both labels.

## Tie-breakers

1. When a sentence is genuinely ambiguous between two labels, prefer the label
   whose *do-not-assign* list does not exclude it; if still tied, assign both
   only if each label's positive criterion is independently met.
2. Negations ("we do not currently use AI") assign no label.
3. Quantitative disclosure without qualitative claim ("AI-related capex was
   £40m") → adoption only if the spend is tied to the company's own
   deployment/development.
