# What Is a Simulated Respondent Worth?

Code and results for our paper, *What Is a Simulated Respondent Worth? Stretching Survey Budgets with Language Models*.

People increasingly ask language models to answer surveys in place of real respondents. We wanted a plain answer to how much that is worth, so we count it in people: a model is "worth 25 respondents" if its answers are as accurate, on average, as a random sample of 25.

We ran sixteen models on 961 older survey questions and on about 450 poll questions fielded after the models were trained. In short:

- A single model is worth somewhere between 6 and 49 respondents, depending on the model and the questions.
- Adding more models barely helps, because they tend to get the same things wrong. All sixteen together are worth 25 respondents on the older surveys. The best one alone is worth 23.
- Showing a model real answers from the same survey helps much more. For the strongest model, the twelve others run on the same polls add 1.5 respondents, and the results of three other questions in the survey add 7.

We developed all of this on five weekly poll reports, then ran the finished protocol once on the next two, which we had kept aside. The three results held.

## Reproducing the paper

You don't need a GPU or any API keys for this. We don't redistribute the survey data, so you download it from the original publishers and our script converts it. You'll need Python 3.12 and `pdftotext` (`apt install poppler-utils` on Ubuntu). The analysis took us about 35 minutes on four CPU threads.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. get the original files; data/download_list.tsv has every name and link
mkdir -p data/raw
while IFS=$'\t' read -r name url; do
  curl -fL -A "Mozilla/5.0" -o "data/raw/$name" "$url"
done < data/download_list.tsv

# 2. convert them into the format the analysis reads
python scripts/prepare_data.py

# 3. unpack our saved model answers
tar xzf simulated-respondent-worth-model-answers.tar.gz

./reproduce.sh
```

You can also download the files by hand and drop them into `data/raw/`. PDFs and other files with a checksum can keep whatever name they came with. `python scripts/prepare_data.py --list` shows what is still missing.

The script checks each file against the version we used and stops if one has changed. If a publisher has replaced a file, `--allow-changed` lets you continue, but the numbers may then differ from the paper.

Results go to `results/`, and tables and figures go to `generated/`. At the end, `verify.py` checks them against the numbers in the paper (kept in `expected/`). If everything lines up, the last line says `all match`.

## Finding your way around

The script names are the ones we used while working, so they aren't very descriptive. This is where to look for each part of the paper:

| Part of the paper | Scripts in `src/` |
|---|---|
| What each model, pair and set is worth (Sections 4, 5) | `y_cov.py`, `z_nest.py` |
| Best non-negative weights for a set of sources (Appendix A) | `nnweights.py` |
| How worth varies across single questions (appendix) | `w_perq.py` |
| Showing other questions from the same survey (Section 6.1) | `w_groups.py`, `w_score.py` |
| Controls with other pollsters and the previous week | `w_ctl.py` |
| Other pollsters and older surveys in the prompt | `w_rel.py`, `w_old.py` |
| Which questions to field, a fixed interview budget | `w_design.py`, `v_thin.py`, `v_alloc.py`, `v_transfer.py`, `w_budget.py` |
| Estimates for groups (Section 6.2) | `w_score.py`, `w_extra.py`, `w_free.py` |
| The same question asked earlier (Section 6.1) | `w_wave.py`, `z_wave.py` |
| Extra group data, anchors, panels, one question per call | `z_groups.py`, `z_anchor.py`, `z_panel.py`, `z_single.py` |
| More models or real answers (Figure 2a) | `w_curve.py` |
| The ceiling on any average of models | `w_ceiling.py` |
| Why overlapping errors hurt averaging and help anchoring (Equation 5) | `w_overlap.py` |
| Using a worth taken from other surveys | `w_unknown.py` |
| Comparison with the coverage-based effective sample size | `w_cover.py` |
| Statistical alternatives (learned corrections, small-area estimates) | `w_strong.py` |
| One question per call | `w_one.py` |
| The held-out replication | `w_fresh.py` |
| Tables | `w_figs.py`, `z_figs.py` |
| Figures 1 and 2 | `paper_figs.py` |

## Asking the models again

You only need this if you want new model answers. It costs money or quota, and models change over time, so don't expect to get exactly our numbers back. The commands we used for each experiment are in `scripts/collect/`; `run_late.sh` and `run_checks.sh` there cover the last ones.

GPT and Claude models can be called in two ways, chosen with `SOL_BACKEND`:

- `cli` is the default and what we used for the paper. It needs `codex` and `claude` installed and logged in.
- `api` calls the OpenAI and Anthropic APIs directly. Run `pip install -r requirements-api.txt` and set `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

```bash
SOL_BACKEND=cli SOL_MODEL=gpt-6.1-sol SOL_SUFFIX=_gpt-6.1-sol python src/x_ladder.py 50 4 medium
SOL_BACKEND=api SOL_MODEL=claude-opus-5-5 SOL_SUFFIX=_claude-opus-5-5 python src/x_ladder.py 50 4 medium
```

Both send the same prompt and answer format with tools turned off. None of the results in the paper came through `api`, and the command-line clients add request settings of their own, so the two can give somewhat different answers.

Saved answers are never overwritten. To collect one again, move or delete its folder in `outputs/sol/`. For open models, start a local OpenAI-compatible server and set `SOL_MODEL=local:<port>:<name>`.

## Data

This repository doesn't contain any survey data. `data/download_list.tsv` lists where to get it, and `data/polls_2026/recipe.json` records where each question and number sits in the original documents. Please follow each publisher's terms when you use their files; the sources are listed in `data/README.md`. Our saved model answers are in `simulated-respondent-worth-model-answers.tar.gz`.
