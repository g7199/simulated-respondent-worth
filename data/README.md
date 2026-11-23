# Data

Download the files in `download_list.tsv` into `data/raw/`, then run `python scripts/prepare_data.py`. This repository does not include the files themselves.

| Built file | Comes from | Terms |
|---|---|---|
| `simbench/SimBenchPop.csv` | [SimBench](https://huggingface.co/datasets/pitehu/SimBench) | CC BY-NC-SA 4.0 |
| `gss/` | [General Social Survey](https://gss.norc.org/), 1972 to 2024 cumulative file | NORC at the University of Chicago |
| `polls_2026/items.json` | 250 questions from public reports by The Economist/YouGov, YouGov UK and EuroTrack, AP-NORC, Reuters/Ipsos, NPR/PBS News/Marist, Gallup, Angus Reid Institute and Essential Research | Each pollster's own terms |
| `polls_2026/econ_*.json` | The five weekly Economist/YouGov crosstab reports, and the next two, kept aside for the replication (`econ_fresh.json`) | YouGov's terms |

`polls_2026/recipe.json` records where each question, answer option and number appears in the source documents. It holds positions, not poll results. About 6% of the question and option text could not be matched to the source documents, mostly the wording of Essential Research and Angus Reid questions, so that part is stored as text.

We only use published aggregate results, with no individual-level poll responses. The GSS file is public microdata, and we use it to compute group shares.

The model answers archive in the repository root holds what the models returned, stored by question id. The annotator prompts come from [Community Alignment](https://arxiv.org/abs/2507.09650) (CC BY 4.0), which you only need if you collect those answers again.
