---
title: "Dormant: an open-weight model read my Mac's Spotlight logs and found 71 GB I will never open again"
published: 
tags: devchallenge, hf26challenge
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge: Week 3: Slow and Steady](https://dev.to/challenges/hacktoberfest-week3-2026-10-19)*

## What I Built

Dormant answers one question about a file on my Mac: *is this ever coming back?*

Not "is this file important", which is a different question and a much harder one. Just: this file has been shut for three weeks, will it be opened again inside ninety days? If the answer is no, it goes on a list. If I approve the list, Dormant moves those files out of the way.

It offers three levels of letting go, and the ladder is deliberately uneven. **Stage** moves files into a dated folder that `undo` reverses from the index. **Trash** moves them to the macOS Trash, so Finder's own Put Back still works. **Delete permanently** is the only path that calls `unlink`, it requires the caller to supply the exact word `DELETE`, it writes the manifest of what is about to be lost *before* the first file goes, and `undo` refuses to touch a delete manifest. The first two are also tested against a tripwire: `os.remove`, `os.unlink` and `os.rmdir` are patched to raise, so if they ever learn to delete, the suite fails.

Before any of that, there is a preview. Selecting six rows and pressing `view` shows the actual images, PDFs, text and video in a drawer, because the fastest way to trust a list of 519 files is to look at six of them and see that it is right.

On the machine I built it on, the answer is that 519 files totalling 71.0 GB are not coming back. Most of it is one drone footage folder, 68.2 GB, that I have not opened once since the week I filled it.

The effort it saves is the specific kind of effort I keep postponing. I know exactly which folder is the problem. I have opened it eleven times, looked at 400 file names, felt the cost of deciding about each one, and closed it. That decision is not a memory problem, it is a *thousand small decisions* problem, and no human who has a deadline is going to make them on a Tuesday.

So the sloth part is literal. Dormant makes the thousand decisions overnight, in a form I can approve with one keypress, and it is wrong about 19 of them, which I can undo.

![The dormancy field](https://raw.githubusercontent.com/soumwadeep/Dormant/main/docs/screenshots/dormancy-field.png)

## Demo

The dashboard is a local FastAPI service plus one page of vanilla JS. It runs entirely on the machine, and the three.js build is vendored into the repository rather than loaded from a CDN, because a product whose whole argument is "this never needs the internet" cannot afford to depend on one.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python cli.py scan            # index files from stat() and Spotlight metadata
python cli.py brief           # the two-line morning brief
python cli.py bench           # the scoreboard, on your machine
python cli.py serve           # http://127.0.0.1:8712
```

The brief, from my actual machine:

```
Dormant found 519 files worth 71.0 GB it is 99% sure you will not reopen
inside 90 days. On the held-out slice it was right 99% of the time it said stop.
```

Every number in that sentence is computed. The model is never asked to restate data, because a language model that rounds 71.3 GB to "about 70" is a bug, not a feature.

There is a synthetic corpus for screenshots and demos, because the index contains file paths, sizes and the dates I opened things, and that is not something I want in a public repository:

```bash
python cli.py --demo demo && python cli.py --demo serve
```

## Code

https://github.com/soumwadeep/Dormant

## How I Built It

The core is [TabPFN](https://github.com/PriorLabs/TabPFN), Prior Labs' open-weight tabular foundation model. It is not a language model doing arithmetic. It is a transformer trained once, offline, on millions of synthetic tabular datasets, which then does in-context inference: your labelled rows go in as context and a posterior comes out. No gradient step on my data, no training run, no per-token bill.

**The labels were already on my machine.** This is the part that made the project possible in a week. macOS records every date a document app opened a file. `mdls -name kMDItemUsedDates` returns the whole list, and `kMDItemLastUsedDate` returns the last one. So I did not have to run a logger for a month and hope. I read three years of my own file-open history on the first pass.

For a file created at date `C` with recorded opens `d1..dk`, Dormant builds one observation row per sampled point inside each dormancy streak. A row's features are only ever what existed at that date: age, days since last open, days since modification, how many times it has been opened, mean and max gap between opens, size, extension, folder depth, weekday and hour. The label is:

- `1` if some later open falls within 90 days of the observation date
- `0` if the next open is more than 90 days away, or never arrives, and the window has closed
- **dropped** if the 90-day window is still open, because the outcome has not happened yet

That third clause is the difference between an evaluation and a fiction. A window that has not closed is censored, and Dormant throws the row away rather than guessing.

The split is on the observation date, never random. Two rows from the same file share a history, so a random split puts yesterday inside tomorrow. Validation is rolling-origin: train on all history up to a cut, score the slice immediately after it.

Two passes for ingestion, deliberately. `os.walk` plus `os.stat` gives the whole universe in 7 seconds. `mdls` is a subprocess round trip to the Spotlight daemon, so it only runs on personal files, in a thread pool, 21 seconds. My first version paid 180 seconds to ask 8,000 repository files whether they had been opened, which was both slow and wrong, because nobody double-clicks a vendored PHP file and a model trained on those rows learns that everything in a repository is dead weight. Files under `.git`, `node_modules`, `vendor`, `Pods` or any build-system marker are indexed but never scored.

The disk-fill forecast is arithmetic, not a model. Free space comes from `disk_usage()`, the growth rate from a Theil-Sen fit over nightly snapshots, and the projection is a line. TabPFN is offered a second, learned projection of the same series, because Prior Labs describes the product as forecasting, but the headline date is always the one you could check with a calculator.

One security decision worth stating, because previewing files is a new kind of exposure: the server generates a token at startup and injects it into the page's own HTML, and every endpoint that returns file *contents* requires it. Anything else that can reach `127.0.0.1:8712`, another process, or a website your browser visits while the dashboard is open, gets a 403. There is a test for that too.

![Model bench and register](https://raw.githubusercontent.com/soumwadeep/Dormant/main/docs/screenshots/model-bench.png)

### What it measured

Real index on this machine: 88,250 files walked in about 12 seconds, 12,719 openable files stored, 669 personal documents, 283 of them carrying recorded open history, 6,918 labelable observation rows. Time-split holdout of 4,842 train and 1,800 test rows, cut 2025-03-01, 10.6% positive.

| model | AUC | precision | recall | GB reclaimed | wrong calls |
|---|---|---|---|---|---|
| rule: 90 days idle | 0.378 | 0.862 | 0.704 | 3.9 | 181 |
| logistic regression | 0.821 | 0.959 | 0.714 | 45.0 | 49 |
| random forest | 0.920 | 0.985 | 0.758 | 44.4 | 19 |
| HistGradientBoosting | **0.962** | 0.988 | 0.895 | **45.3** | 18 |
| TabPFN-2 (open weights, what ships) | 0.956 | 0.987 | 0.885 | 45.1 | 19 |

The first row is the one I actually learned from. "Archive anything untouched for ninety days", which is what I would have written by hand in an afternoon, scores **worse than a coin flip**. Ninety days idle is exactly what a file looks like in the week before a deadline drags it back. The model is not beating a strawman. It is beating the heuristic that motivated the product, and that heuristic was mine.

I also benchmarked the models as the labelled corpus shrinks, because a personal corpus is small and that is the regime an in-context model is supposed to own. Here is the part I expected to be a win and it is not:

| labelled rows | TabPFN | hist gb | random forest | logistic |
|---|---|---|---|---|
| 120 | 0.934 | **0.936** | 0.883 | 0.654 |
| 240 | 0.944 | **0.961** | 0.906 | 0.716 |
| 480 | 0.943 | **0.952** | 0.885 | 0.935 |
| 960 | 0.943 | **0.956** | 0.885 | 0.775 |
| 1,920 | 0.947 | **0.953** | 0.902 | 0.801 |

HistGradientBoosting beats the open-weight model at every corpus size on my machine. It still ships as the default, for reasons that are not the last decimal: no training run, no feature engineering loop, no hyperparameters, 444 MB of RAM, and changing the question from 30 days to 90 is a constant and a re-run rather than a re-fit. The baselines stay in the scoreboard precisely so that claim can be checked, and this table is what happened when I checked it.

43 tests, mostly about the two things that could actually hurt someone: label leakage, and what happens to a file when you press the button.

## Why Does Open Innovation Matter?

**There is no upload path.** The training set is my Spotlight database: every date I opened every file on this machine. A closed API cannot have that, not because of a policy I would object to, but because the product would have to ship my filesystem's behavioural history to a server to do something that takes 444 MB of RAM locally. The privacy argument here is not a promise. It is an absence.

**Per-user adaptation costs nothing and happens every run.** The model is fitted to *this* machine's habits each time, in 0.3 seconds, because in-context inference is just putting the rows in the prompt. A hosted model would need my history on its infrastructure. A fine-tuned model would need a GPU and a training run. Neither is what a personal utility should ask for.

**It fits the laptop that has to run it.** This is an 8 GB M1 with about 65 GB free, which is the actual constraint the whole project lives inside. TabPFN-2's checkpoint is 28 MB and peaks at 444 MB of RSS. I tested the recommended TabPFN-3.5 first: its fast checkpoint is 334 MB, the full model 876 MB, and the package puts a licence-acceptance wall plus an account in front of it. So 3.5 is wired in behind one environment variable, `DORMANT_TABPFN_MODEL=3.5-fast`, and is not the default. A product about doing nothing should not require a signup.

**Cost to run is zero, permanently.** 2,500 predictions in about 10 seconds on CPU. No meter. I can run it nightly for three years and it will never be a bill.

**Swapping the model is a config change, not a rewrite.** `dormant/model.py` treats the predictor as a provider and the baselines sit in the same scoreboard on purpose, so the open-weight claim is falsifiable rather than decorative. HistGradientBoosting does currently edge TabPFN on AUC on this corpus, 0.967 against 0.955, and I would rather say that here than have someone find it in the code.

**Where open worked better than closed:** the moment I wanted to change what "dormant" means, from 30 days to 90, the whole thing stayed mine. That is a horizon constant and a re-run. With a hosted model I would be re-deciding what I was willing to upload each time I changed my mind.

## My Agent Session

The engine, the dashboard and this post were built in one long agent session against a live index of my own machine, which is the fastest way I have found to check a claim: the numbers in this post came from a terminal, not from a paragraph I wrote first.

The two most valuable minutes were the go/no-go test before any architecture existed. Install TabPFN, load the smallest checkpoint, predict 3,000 rows, watch peak RSS. If that had needed 4 GB, there would be no project, and I would have known in five minutes rather than five evenings.

The most instructive failure was the operating point. My first benchmark reported TabPFN at AUC 0.33, which I briefly believed about my own model. It was an orientation error: the model predicts *will reopen*, the product decides *is dormant*, and I had scored the complement as if it were the class. Fixing it also exposed a second trap, which is that with 91% of rows dormant, a threshold of 0.03 "meets 95% precision" by flagging everything. That is not a decision, it is a description of the majority class, and there is now a test that refuses thresholds below 0.5.

*(DevRelay session embed goes here.)*

## The Lazy Payoff

I ran it on my own machine, so these are my numbers, not a projection of them.

- **519 files, 71.0 GB** identified as not coming back, 68.2 GB of it in one folder.
- **One decision instead of about a thousand.** The folder I have been avoiding is `DroneShots`, 68.2 GB, which I have opened maybe four times since I filled it.
- **19 wrong calls on a 1,800-row holdout**, and on the live list every one of them is reversible, because staging and Trash write a manifest and `undo` puts each file back at its exact original path. Permanent delete needs the word DELETE typed and refuses to be undone.
- **12 seconds** for a full scan of 88,250 files, which is roughly the time it takes me to decide not to do this myself.
- **$0 and 0 API calls**, forever, on a laptop that cannot run the alternative anyway.

The honest part: only 283 of my 669 personal documents carry any recorded open history at all. The other two thirds are scored from age, size, type and folder alone, which is why the product is named after an animal that is in no hurry. Every night it runs, it converts more of that population from unknown to observed, and the cold start stops being a cold start. The disk-fill projection is the same story: it needs two distinct days of snapshots before it will give you a date, and on a first run it says "no measurable fill rate" instead of inventing one.

## Prize Categories

**Best Use of TabPFN.** TabPFN is the product, not a dependency. The open-weight tabular foundation model is the thing that decides whether 71 GB of my disk is a library or a landfill, it runs locally on CPU in 444 MB of RAM with a 28 MB checkpoint, and it is benchmarked against logistic regression, a random forest, HistGradientBoosting and the naive 90-day rule on a leak-free time-split holdout of my own file history. The recommended TabPFN-3.5 is supported behind one environment variable and documented, with the reason it is not the default stated plainly: 334 MB of weights and an account wall on an 8 GB laptop defeats the point.

---

Thanks for reading. If you want to run it against your own disk, the repo has the four commands above and the tests will tell you whether the numbers reproduce on your machine, which they may not, because the model is fitted to whatever history your Mac happens to have kept.
