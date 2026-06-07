# Patrick's notes — "MRM final todos" + manuscript pass (verbatim)

Captured 2026-06-07. These are Patrick's own notes, kept as written (typos included)
so nothing is lost in paraphrase. Cross-referenced and themed in
`inline_comments_inventory.md`.

## MRM final todos / Patrick comments manuscript pass

- **Fig7 (Fisher):** get rid of correlation description subplot a) unnecessary, update y axis of subplot b, make more descriptive and more concise (no parenthesis), get rid of bayesian crab title over title of subplot b as well, I wonder if we could not rearrange the subplots and make two rows but put the third in the middle on the x-axis so that we have 2 subplots in row 1 and 1 in row 2 that is situated in between the upper subplots, this way we can make the plots bigger and match the font sizes of other figures for axes descriptions and titles and also the legend need have bigger fonts without getting a width that is a lot bigger than two subplots together, also I find the current description of the factors between the unconstraiend CRLB, bayesian CRLB and NUTS posterior unceraitny a bit vague and not so easy to understand when listed in the figure description, maybe we could put them back in the figure for both comparisons (unconstriaend to bayesian and then to NUTS)…we could maybe put them in the respective color above the bars for both cases? If we have a new arrangement of the subplot as described above we could maybe also bring the SNR descriptions of the horizontal noise lines back inside the plot to make legend above a bit less crowded

- **Fig 8 (validation):** is currenlty a bit to big for one page, so that the bottom right corner is hidden by page number and also the entire fig descriptions vanishes and is cut off … not sure how to best handle this … maybe make a tiny bit smaller and put the fig8 description on next page? Not sure how to do this in latex …

- **Table1:** I wonder if we should not include the AUCs for all the individual features as well (all that are also visualised in fig 2 for reference), what are DeLongs tests? Are they still up to date and used?

- **Supplementary Atlas** is not included in paper currently and the description is wrongly formatted with current "D=…" description … also I would like to have encoding directions first and then the ROIs … with both supplementary figures I have same problem as with fig 8, the are so big that the figure description is cut off at the end … need to find a way to respect end of page

- **Fig6 (uncertainty):** make a bit smaller maybe so that figure description fits on page, currenlty moves into page number

- Put references and citations review as last item on project state todo

- Check if we can cite Obsidian if not already done

- In general **Stephan** mentioned that I dont need to make as many sub titles, like "MAP Estimation" or "Bayesian Formulation" … can just be different paragraphs

- Maybe copy paste the latest author guidelines from MRM site to double check that final draft fulfils formal requirements

- **Manuscript prose:** not sure if it is because we have different sections in different files but we are kind of repetitive throughout the manuscript regurgitating a lot of already said things, lets make sure it is concise and not to redundant

## Bigger conceptual items (from the conversation, 2026-06-07)

- **Uncertainty story.** Hopes of using posterior std (per-component spectral uncertainty)
  as a tumor-confidence signal for an "uncertainty-aware classifier" look weaker than hoped.
  Two probes tried: (1) misclassified vs correctly-classified posterior std; (2) ROIs far
  from vs close to the decision boundary. The 2nd is largely **logistic-regression geometry**.
  New thought: that geometric uncertainty may still be *usable* (a real predictive confidence),
  just not Bayesian-special — you'd get it from a point estimate too. Still want to test whether
  the **full posterior** adds "signs of life" beyond the point estimate. Identifiability story
  (std → which bins are reliable) is solid; question is whether it **propagates** to the
  classification task and makes a difference.

- **Restructure.** Patrick previously proposed reordering figures/sections into thematic
  blocks (see inline comment at results.tex:57). Discuss step by step.

- **Fig 7 (Fisher) rework.** See Fig7 bullet above.

- **Approach:** go slow and deliberate, focus on the hard stuff, step by step. Yesterday's
  over-ambitious rewrite had to be reverted — don't repeat that.
