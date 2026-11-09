# Run on an NVIDIA server

Generative models such as SmolVLM need an NVIDIA graphics card. This guide takes an experiment prepared on your computer, runs it on a server, and brings the report back.

Start from an experiment folder made with the [tutorial](tutorial.md#4-set-up-your-own-images). In this guide it is called `my_experiment`, and it sits next to the `VLMShift` folder.

> **Not yet tried with a real card.** Steps 2 to 4 were replayed word for word on a fresh Linux machine with no graphics card, and the fixes it needed are written in. SmolVLM then answered through VLMEvalKit on the processor, after changing one line of VLMEvalKit for that test only. Step 4 exists to show what works on your server in a few minutes.

## 1. What you need

| You need | How to check |
|---|---|
| A Linux server with an NVIDIA card | `nvidia-smi` lists the card |
| Python 3.11 | `python3.11 --version` |
| A C compiler: one package must be compiled | `gcc --version` |
| Git: VLMEvalKit is installed from its repository | `git --version` |
| About 20 GB of free disk for the libraries and a small model | `df -h ~` |
| Internet access on the server, for the first download | `curl -I https://huggingface.co` |

## 2. Copy your work to the server

On your computer, from the folder that holds `VLMShift` and `my_experiment`, make one archive. Replace `user@server` with your own login:

```bash
tar --exclude=.venv --exclude=workspace -czf vlmshift.tar.gz VLMShift my_experiment
```

```bash
scp vlmshift.tar.gz user@server:
```

On the server, unpack it:

```bash
tar -xzf vlmshift.tar.gz
```

The paths in `table.csv` are relative, so there is nothing to change.

## 3. Install

On the server, create an environment and install everything:

```bash
cd VLMShift
```

```bash
python3.11 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -e .
```

```bash
pip install -r environments/vlmevalkit.txt
```

This installs about 220 packages and takes about 7 GB. On a test Linux machine it took 31 minutes. On Linux, the default `torch` is already built for NVIDIA cards.

A server usually has no screen, so swap OpenCV for its server version. Without this, VLMEvalKit cannot load and asks for `libGL.so.1`:

```bash
pip uninstall -y opencv-python
```

```bash
pip install opencv-python-headless
```

pip then warns that `vlmeval` requires `opencv-python`. This is expected: both versions provide the same module.

## 4. Check before the long run

Go to your experiment and try every model:

```bash
cd ../my_experiment
```

```bash
vlmshift check experiment.yaml
```

The tool shows the graphics card, asks each model up to four real questions, and estimates the length of the full run. No answer is kept, so the check changes no result. The first time, it downloads the model: about 500 MB for `SmolVLM-256M`.

With a fake model, on a computer with no card, it looks like this:

```text
Graphics cards: none seen by torch
Libraries: python 3.14.2, torch 2.12.0, transformers 5.22.2
demo (demo)
  Loaded in 0 s
  Question: Y a-t-il un cercle dans cette image ? Répondez par oui ou non.
  Answer:   'non'
  Read as:  no (0.0 s)
  Question: Is there a circle in this image? Answer yes or no.
  Answer:   'No.'
  Read as:  no (0.0 s)
  All 8 variants: about 0 s
Next: read the answers above. If they look right, start the run.
```

Read each answer yourself. The check proves that a model responds, not that it is right.

| You see | What it means | What to do |
|---|---|---|
| `Graphics cards: none seen by torch`, then `Found no NVIDIA driver` | `torch` cannot use the card | Run `nvidia-smi`. The default `torch` needs a driver for CUDA 13: the first line of `nvidia-smi` shows the CUDA version. If yours is older, install the `torch` that matches it from [pytorch.org](https://pytorch.org/get-started/locally/) |
| `FAILED: ImportError: libGL.so.1` | OpenCV looks for a screen | Do the OpenCV swap at the end of [step 3](#3-install) |
| `FAILED: ModuleNotFoundError` | This model needs one more package | Install the package named in the message |
| `FAILED: ImportError` or `AttributeError` | This model needs another `transformers` | See [other models](#7-other-models) |
| `FAILED:` with `401` or `gated` | The model asks you to accept its terms | Accept them on its Hugging Face page, then run `hf auth login` |
| `FAILED:` with `out of memory` | The card is too small, or already in use | Run `nvidia-smi` to see who uses it. [Choose another card](#choose-a-card) or a smaller model |
| `Read as:  NOT UNDERSTOOD` | The reader found neither yes nor no | Add the words of the answer to `task.answers`. If the answer is cut, raise `inference.max_new_tokens` |
| `! model_note` | A setting may not be applied by this model | See the [notes on settings](documentation.md#generative-models-through-vlmevalkit) |
| `Did not detect the .env file` | VLMEvalKit looks for API keys | Nothing: local models do not need them |

The command ends with an error if a model gave no answer, so it can stop a script before a long run.

## 5. Start the run

Start it so that it survives when you log out:

```bash
nohup vlmshift all experiment.yaml analysis.yaml > run.log 2>&1 &
```

To follow it, read the log. `Ctrl+C` stops the reading, not the run:

```bash
tail -f run.log
```

Each model prints one line per minute:

```text
  smolvlm: 240 of 2000, 0 errors, about 52 min left
```

If the errors climb from the start, stop the run and read the messages:

```bash
pkill -u "$USER" -f "vlmshift all"
```

If the run stops for any reason, start the same command again. Answers already obtained are kept, and only the missing ones are asked.

### Choose a card

To use the second card of the server, start the line with its number. Cards are numbered from 0:

```bash
CUDA_VISIBLE_DEVICES=1 nohup vlmshift all experiment.yaml analysis.yaml > run.log 2>&1 &
```

On a shared cluster with Slurm, put `vlmshift all experiment.yaml analysis.yaml` in your job script. Your administrators can tell you how to ask for a card.

## 6. Bring the results back

The last line of `run.log` gives the path of `report.md`. Copy its whole folder to your computer, replacing the path with yours:

```bash
scp -r user@server:my_experiment/workspace/runs/RUN/analyses/ANALYSIS/report .
```

Open `report.md`, then see [read your results](tutorial.md#7-read-your-results). To change the analysis, edit `analysis.yaml` on the server and run the same command again: no model is called.

## 7. Other models

Add a line under `models` in `experiment.yaml`. The `id` is the name of the model in the [VLMEvalKit catalogue](https://github.com/open-compass/VLMEvalKit/blob/3c192143007e4eb3f5c1a36558a4f8c1451980cb/vlmeval/config.py):

```yaml
models:
  - {name: smolvlm, adapter: vlmevalkit, id: SmolVLM-256M}
  - {name: smolvlm-500, adapter: vlmevalkit, id: SmolVLM-500M}
```

VLMEvalKit recommends a [`transformers` version for each family](https://github.com/open-compass/VLMEvalKit/blob/3c192143007e4eb3f5c1a36558a4f8c1451980cb/README.md). If a family needs another version, give it its own environment, then write its name in place of `VERSION` and `MODEL`:

```bash
deactivate
```

```bash
python3.11 -m venv ../VLMShift/.venv-other
```

```bash
source ../VLMShift/.venv-other/bin/activate
```

```bash
pip install -e ../VLMShift -r ../VLMShift/environments/vlmevalkit.txt "transformers==VERSION"
```

Then do the OpenCV swap at the end of [step 3](#3-install) in this environment too.

```bash
vlmshift check experiment.yaml --only MODEL
```

```bash
vlmshift run experiment.yaml --only MODEL
```

All the models fill the same run, so the report compares them.

## 8. For a result you want to publish

**Fix the weights.** Write the `revision` of each model. Without it, the weights can change between two runs:

```yaml
models:
  - {name: smolvlm, adapter: vlmevalkit, id: SmolVLM-256M, revision: 7e3e67edbbed1bf9888184d9df282b700a323964}
```

**Keep the versions.** Each run records its libraries and models in `run_log.json`, in the run folder.

## If the disk or the network is limited

| Situation | What to do |
|---|---|
| `pip install` stops on `polygon3` with `Python.h` | Ask your administrators for the Python headers (`python3.11-dev` on Ubuntu) |
| Little space in your home folder | Run `export HF_HOME=/path/with/space` before any command. The weights go there |
| The card has no internet access | On a machine that has it, run `hf download HuggingFaceTB/SmolVLM-256M-Instruct` with the same `HF_HOME`. Then run `export HF_HUB_OFFLINE=1` before starting |
