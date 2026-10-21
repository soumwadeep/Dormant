---
title: "Dormant: an open-weight model read my Mac's Spotlight logs and found 71 GB I will never open again"
published: 
tags: devchallenge, hf26challenge
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 3: Slow and Steady](https://dev.to/challenges/hacktoberfest-week3-2026-10-19)*

## What I Built

**Dormant answers one question about a file on my Mac: is this ever coming back?**

Not "is this file important", which is a different and much harder question. Just this: the file has been shut for three weeks, will it be opened again inside ninety days? If the answer is no, it goes on a list. If I approve the list, Dormant moves those files out of the way.

The core is [TabPFN](https://github.com/PriorLabs/TabPFN), an open-weight tabular foundation model, running locally on CPU. No internet, no API key, no account, no bill. It was built and benchmarked on an 8 GB Apple M1, which is the kind of machine most of us actually have.

### The chore it removes

I know exactly which folder is the problem. I have opened it eleven times, looked at four hundred file names, felt the cost of deciding about each one, and closed it. That is not a memory problem. It is a *thousand small decisions* problem, and no human with a deadline is going to make them on a Tuesday.

So the sloth part is literal. Dormant makes the thousand decisions overnight, in a form I can approve with one keypress.

### What it found on my machine

```
Dormant found 519 files worth 71.0 GB it is 99% sure you will not reopen
inside 90 days. On the held-out slice it was right 99% of the time it said stop.
```

68.2 GB of that is one drone footage folder I have not opened since the week I filled it.

### Three levels of letting go

The ladder is deliberately uneven, because the destructive rung should be harder to reach than the recoverable ones.

| action | what happens | reversible by |
|---|---|---|
| **Stage** | moves files into a dated folder, writes a manifest | `undo`, from inside Dormant |
| **Trash** | moves files to the macOS Trash | Finder's Put Back, or `undo` |
| **Delete permanently** | the only path that calls `unlink` | nothing in this program |

Permanent delete requires the caller to supply the literal word `DELETE`, writes the manifest of what is about to be lost *before* the first file goes, and `undo` refuses to touch a delete manifest. Stage and Trash are tested against a tripwire: `os.remove`, `os.unlink` and `os.rmdir` are patched to raise, so if they ever learn to delete, the suite fails.

And before any of it, there is a preview. Selecting six rows and pressing **view** shows the actual images, PDFs, text and video in a drawer, because the fastest way to trust a list of 519 files is to look at six of them and see that it is right.

![The dormancy field](https://raw.githubusercontent.com/soumwadeep/Dormant/main/docs/screenshots/dormancy-field.png)

## Demo

The dashboard is a local FastAPI service plus one page of vanilla JavaScript. The three.js build is vendored into the repository rather than loaded from a CDN, because a product whose whole argument is "this never needs the internet" cannot afford to depend on one.

### Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python cli.py scan      # index files from stat() and Spotlight metadata (~12s)
python cli.py brief     # the two-line morning brief
python cli.py top       # the files it thinks are dead
python cli.py bench     # the scoreboard, on your machine
python cli.py serve     # mission control on http://127.0.0.1:8712
```

### See it without touching anyone's disk

The index contains file paths, sizes and the dates I opened things. That is personal, so the repository ships a synthetic corpus with the same shape and none of the names. Every screenshot in this post came from it.

```bash
python cli.py --demo demo && python cli.py --demo run --bench && python cli.py --demo serve
```

### What the dashboard shows

- **Morning brief.** Two lines. Every number in them is computed, never generated.
- **Dormancy field.** A 3D plot where radius is a file's age and height is the model's dormancy score, so the shape of a neglected disk is visible before you read anything.
- **Volume forecast.** A gauge, the growth curve, and the date the volume reaches a warning level.
- **Reclaim ledger.** Gigabytes identified, files on the stop list, days bought at the current growth rate.
- **Model bench.** The scoreboard below, plus a chart of how each model degrades as the labelled corpus shrinks.
- **Dormant register.** The ranked list: filterable, sortable, multi-selectable, with preview, stage, trash and delete on the selection.
- **Agent console.** One sentence in, validated tool calls out.

![Model bench and register](https://raw.githubusercontent.com/soumwadeep/Dormant/main/docs/screenshots/model-bench.png)
![Light theme, cyan accent](https://raw.githubusercontent.com/soumwadeep/Dormant/main/docs/screenshots/light-cyan.png)
![Mobile at 390px](https://raw.githubusercontent.com/soumwadeep/Dormant/main/docs/screenshots/mobile.png)

## Code

https://github.com/soumwadeep/Dormant

GPL-3.0. Six commits, 43 tests, no build step.

```
dormant/
  scanner.py   two-pass ingestion: os.walk for the universe, mdls for history
  features.py  observation points inside each dormancy streak, and their labels
  dataset.py   time-split holdout and rolling-origin folds
  model.py     TabPFN provider, four baselines, the scarcity benchmark
  forecast.py  disk projection from measured snapshots or a reconstructed curve
  brief.py     the two lines, as a template over computed integers
  action.py    stage, trash, permanent delete, undo, preview metadata
  agent.py     a sentence in, validated tool calls out
  demo.py      the synthetic corpus
api/app.py     FastAPI service, bound to 127.0.0.1
cli.py         scan, brief, top, bench, run, preview, archive, trash, purge, undo, serve
web/           one page, vendored three.js, two themes, four accents, mobile tabs
tests/         43 tests: label leakage, evaluation integrity, file safety
```

## How I Built It

### The model, and why it is the right shape for this

TabPFN is not a language model doing arithmetic. It is a transformer trained once, offline, on millions of synthetic tabular datasets, which then does **in-context inference**: your labelled rows go in as context and a posterior comes out. No gradient step on my data, no training run, no per-token bill.

That matters here because a personal file corpus is *small*. This machine produced about seven thousand labelled rows from 669 personal documents. That is precisely the regime an in-context model is built for, and precisely the regime where "let me fine-tune something" is a bad afternoon.

### The labels were already on my machine

This is the part that made the project possible in a week rather than a month. macOS records every date a document app opened a file.

```bash
mdls -name kMDItemUsedDates ~/Documents/tax.pdf
#   "2024-02-11 09:14:02 +0000",
#   "2024-09-27 18:30:00 +0000"
```

So I did not have to run a logger and hope. I read three years of my own file-open history on the first pass.

### Turning that into a supervised problem

For a file created at `C` with recorded opens `d1..dk`, Dormant builds one observation row per sampled point **inside each dormancy streak**. A row's features are only ever what existed at that date: age, days since last open, days since modification, how many times it has been opened, mean and max gap between opens, size, extension, folder depth, weekday and hour.

| label | rule |
|---|---|
| `1` | some later open falls within 90 days of the observation date |
| `0` | the next open is more than 90 days away, or never arrives, and the window has closed |
| **dropped** | the 90-day window is still open, because the outcome has not happened yet |

That third row is the difference between an evaluation and a fiction. An unclosed window is censored, and Dormant throws the row away rather than guessing at it.

The split is on the observation date, never random. Two rows from the same file share a history, so a random split puts yesterday inside tomorrow. Validation is rolling-origin: train on all history up to a cut, score the slice immediately after it.

### Two passes for ingestion

`os.walk` plus `os.stat` gives the whole universe in under two seconds. `mdls` is a subprocess round trip to the Spotlight daemon, so it only runs on personal files, in a thread pool, ten seconds.

My first version paid 180 seconds to ask eight thousand repository files whether they had been opened. That was both slow and wrong: nobody double-clicks a vendored PHP file, and a model trained on those rows learns that everything in a repository is dead weight, which is true and useless. Files under `.git`, `node_modules`, `vendor`, `Pods`, any build-system marker, or any directory that looks like a checkout are indexed but never scored.

### The forecast is arithmetic, not a model

Free space comes from `disk_usage()`. The growth rate comes from a Theil-Sen fit, which is robust to the one day a phone backup lands. The projection is a line. TabPFN is offered a second, learned projection of the same series, because Prior Labs describes the product as forecasting, but the headline date is always the one you could check with a calculator. If the growth rate is below 0.05 GB a day, Dormant says "no measurable fill rate" instead of printing a date eighty years out, which is exactly what my first version did.

### Previewing files is a new kind of exposure

The preview endpoints hand out the *contents* of real files, so they are not open to anything that can reach `127.0.0.1:8712`. The server generates a token at startup and injects it into the page's own HTML; every content endpoint requires it and returns 403 without it. Another process, or a website your browser visits while the dashboard is open, gets nothing. There is a test for that too.

### What it measured

Real index on this machine: 88,250 files walked in about 12 seconds, 12,719 openable files stored, 669 personal documents, 283 of them carrying recorded open history, 6,918 labelable observation rows. Time-split holdout of 4,842 train and 1,800 test rows, cut 2025-03-01, 10.6% positive.

| model | AUC | precision | recall | GB reclaimed | wrong calls |
|---|---|---|---|---|---|
| rule: 90 days idle | 0.378 | 0.862 | 0.704 | 3.9 | 181 |
| logistic regression | 0.821 | 0.959 | 0.714 | 45.0 | 49 |
| random forest | 0.920 | 0.985 | 0.758 | 44.4 | 19 |
| HistGradientBoosting | **0.962** | 0.988 | 0.895 | **45.3** | 18 |
| TabPFN-2 (open weights, what ships) | 0.956 | 0.987 | 0.885 | 45.1 | 19 |

**The first row is the finding.** "Archive anything untouched for ninety days", which is what I would have written by hand in an afternoon, scores *worse than a coin flip*. Ninety days idle is exactly what a file looks like in the week before a deadline drags it back. The model is not beating a strawman. It is beating the heuristic that motivated the product, and that heuristic was mine.

**The second finding is less comfortable, so it is in here.** I benchmarked the models as the labelled corpus shrinks, expecting the open-weight model to take over at small sizes. It does not:

| labelled rows | TabPFN | hist gb | random forest | logistic |
|---|---|---|---|---|
| 120 | 0.934 | **0.936** | 0.883 | 0.654 |
| 240 | 0.944 | **0.961** | 0.906 | 0.716 |
| 480 | 0.943 | **0.952** | 0.885 | 0.935 |
| 960 | 0.943 | **0.956** | 0.885 | 0.775 |
| 1,920 | 0.947 | **0.953** | 0.902 | 0.801 |

HistGradientBoosting wins at every corpus size on my machine. TabPFN still ships as the default for reasons that are not the last decimal: no training run, no feature engineering loop, no hyperparameters, 444 MB of RAM, and changing the question from 30 days to 90 is a constant and a re-run rather than a re-fit. The baselines stay in the scoreboard precisely so that claim can be checked, and this table is what happened when I checked it.

### What went wrong, because that is the useful part

- **An orientation error worth 0.6 AUC.** My first benchmark reported TabPFN at 0.33, which I briefly believed about my own model. The model predicts *will reopen*; the product decides *is dormant*; I had scored the complement as if it were the class.
- **A threshold that was not a decision.** With 91% of rows dormant, a threshold of 0.03 "meets 95% precision" by flagging everything. That is a description of the majority class, not a model. There is now a test that refuses any operating point below 0.5.
- **A stale index.** Rows from an earlier scan were never removed, so file types I had excluded kept appearing in the list. A full scan now prunes what it did not see, but only when the walk completed, so a capped scan cannot delete the half it never looked at.
- **Repo files on the stop list.** Files found through Spotlight were labelled personal even inside a checkout, which is how a vendored asset ended up ranked first. Directory context is now derived while walking, and for index-only files, from the path.

## Why Does Open Innovation Matter?

**There is no upload path.** The training set is my Spotlight database: every date I opened every file on this machine. A closed API cannot have that, not because of a policy I would object to, but because the product would have to ship my filesystem's behavioural history to someone else's server in order to do something that takes 444 MB of RAM locally. The privacy argument here is not a promise. It is an absence.

**Per-user adaptation costs nothing and happens every run.** The model is fitted to *this* machine's habits each time, in 0.3 seconds, because in-context inference is just putting the rows in the prompt. A hosted model would need my history on its infrastructure. A fine-tuned model would need a GPU and a training run. Neither is what a personal utility should ask for.

**It fits the laptop that has to run it.** This is an 8 GB M1 with about 65 GB free, which is the constraint the whole project lives inside. TabPFN-2's checkpoint is 28 MB and peaks at 444 MB of RSS. I tested the recommended TabPFN-3.5 first: its fast checkpoint is 334 MB, the full model 876 MB, and the package puts a licence-acceptance wall plus an account in front of it. So 3.5 is wired in behind one environment variable, `DORMANT_TABPFN_MODEL=3.5-fast`, and is not the default. A product about doing nothing should not require a signup.

**Cost to run is zero, permanently.** 2,500 predictions in about 10 seconds on CPU. No meter. I can run it nightly for three years and it will never become a bill.

**Swapping the model is a config change, not a rewrite.** `dormant/model.py` treats the predictor as a provider, and the baselines sit in the same scoreboard on purpose, so the open-weight claim stays falsifiable rather than decorative.

**Where open beat closed in practice:** the moment I changed what "dormant" means, from 30 days to 90, the whole thing stayed mine. That is a horizon constant and a re-run. With a hosted model I would be re-deciding what I was willing to upload every time I changed my mind.

## My Agent Session

The engine, the dashboard and this post were built in one long agent session against a live index of my own machine, which is the fastest way I have found to check a claim: every number in this post came from a terminal, not from a paragraph I wrote first.

Two moments were worth the whole session.

The first was the go/no-go test before any architecture existed. Install TabPFN, load the smallest checkpoint, predict three thousand rows, watch peak RSS. Ninety seconds later I had the answer: 444 MB, fits, continue. If it had needed 4 GB there would have been no project, and I would have known in five minutes rather than five evenings.

The second was the orientation bug, which only surfaced because the baselines were already in the scoreboard. A model that reports 0.33 AUC next to a logistic regression at 0.82 is telling you something about your code, not about your model. Without the boring baseline sitting in the same table, I would have shipped it.

*(DevRelay session embed goes here.)*

## The Lazy Payoff

I ran it on my own machine, so these are my numbers, not a projection of them.

| | |
|---|---|
| **Files identified as never coming back** | 519 |
| **Space they hold** | 71.0 GB, of which 68.2 GB is one folder |
| **Decisions I made instead of four hundred** | one, plus a preview of six files |
| **Wrong calls, measured on a 1,800-row holdout** | 19 |
| **Full scan of 88,250 files** | 12 seconds |
| **Cost** | $0 and zero API calls, permanently |
| **Recovery if it is wrong** | `undo`, or Finder's Put Back, for anything but a typed DELETE |

The honest part: only 283 of my 669 personal documents carry any recorded open history at all. The other two thirds are scored from age, size, type and folder alone, which is why the product is named after an animal that is in no hurry. Every night it runs, it converts more of that population from unknown to observed, and the cold start stops being a cold start. The disk projection is the same story: it needs two distinct days of snapshots before it will give you a date, and on a first run it says "no measurable fill rate" instead of inventing one.

I have not deleted anything yet. That is deliberate: the list is 519 files, I have previewed six of them, and the folder holding most of the weight is one I am ready to let go of. Staging it tonight, `undo` if I change my mind.

## Prize Categories

**Best Use of TabPFN.** TabPFN is the product, not a dependency. The open-weight tabular foundation model is the thing that decides whether 71 GB of my disk is a library or a landfill. It runs locally on CPU in 444 MB of RAM from a 28 MB checkpoint, it is re-fitted to this machine's habits on every run with no training step, and it is benchmarked against logistic regression, a random forest, HistGradientBoosting and the naive 90-day rule on a leak-free, time-split holdout of my own file history. The recommended TabPFN-3.5 is supported behind one environment variable and documented, with the reason it is not the default stated plainly: 334 MB of weights and an account wall on an 8 GB laptop defeats the point.

---

Thanks for reading. If you want to run it against your own disk, the five commands are above, and `python cli.py bench` will tell you whether these numbers reproduce on your machine. They may not, and that is the interesting part: the model is fitted to whatever history your Mac happens to have kept, so the report is yours.
