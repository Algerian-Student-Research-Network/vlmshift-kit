# Run without an NVIDIA card: Colab and Kaggle

Google Colab and Kaggle lend a graphics card for free, in your browser. VLMShift runs there as on your computer, and a ready notebook runs the demo: `examples/pope_mini/colab.ipynb`.

> **Tested with the versions of Colab, on a processor.** The notebook ran from the first cell to the last with Python 3.13, pandas 2.2.3, torch 2.14 and transformers 5.22, as on Colab. It has not yet run on a real Colab card.

## 1. Open the notebook

| Service | Free card | Good to know |
|---|---|---|
| [Google Colab](https://colab.research.google.com) | One T4, 15 GB | A session lasts a few hours, then everything on it is erased |
| [Kaggle](https://www.kaggle.com/code) | Two T4 of 15 GB, or one P100 | About 30 hours of card per week. A phone number is needed to turn the card on |

On Colab: *File > Open notebook > GitHub*, then paste the address of the project and pick `examples/pope_mini/colab.ipynb`. Or download the file and use *File > Upload notebook*. On Kaggle: *Create > New notebook*, then *File > Import notebook*.

Then choose a card before running anything:

- Colab: *Runtime > Change runtime type > T4 GPU*.
- Kaggle: *Settings > Accelerator > GPU T4 x2*, and *Settings > Internet* on.

Run the cells one by one, from the top.

## 2. What the notebook does

| Step | What happens | Card needed |
|---|---|---|
| 1. Install | Downloads VLMShift. torch and transformers are already installed | No |
| 2. Google Drive | Optional: keeps the workspace on your Drive, so a new session resumes where the last one stopped | No |
| 3. Fetch | 100 photos of COCO and their 1,800 POPE questions, 18 MB | No |
| 4. CLIP | The demo of the [tutorial](tutorial.md#2-run-the-demo), then the comparison with the paper | No |
| 5. InstructBLIP-13B | A model of the POPE paper, then each measure next to the published one | Yes |

InstructBLIP-13B downloads 57 GB the first time. In float16 it needs about 28 GB on the card. On a T4, the notebook shrinks it to 4 bits (`quantize: 4bit`) so that it fits in less than 10 GB. Its answers can then differ a little from the full model. On a card of 32 GB or more, such as an A100 of Colab Pro, the notebook runs the full model, as in the paper.

## 3. If the session stops

A free session can stop at any time: after a few hours, or when you leave it idle. Every answer is written to the workspace as soon as it arrives. If the workspace is on Google Drive (step 2 of the notebook), run the notebook again from the top: VLMShift only asks the questions that have no answer yet. Without Drive, the session's disk is erased and everything starts again.

## 4. Your own experiment

1. Put your experiment folder on Google Drive: the images, the table, `experiment.yaml` and `analysis.yaml`. The [tutorial](tutorial.md#4-set-up-your-own-images) makes them.
2. In a new notebook, install VLMShift as in step 1 of the demo notebook, then mount Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
%cd /content/drive/MyDrive/my_experiment
```

3. Run each step as on your computer, with `!` in front:

```python
!vlmshift check experiment.yaml
```

```python
!vlmshift all experiment.yaml analysis.yaml
```

For the model, the `huggingface` adapter needs nothing more on Colab: see [generative models through Hugging Face](documentation.md#generative-models-through-hugging-face). Models of the VLMEvalKit catalogue need a long installation: follow [step 3 of the server guide](server.md#3-install) in a notebook.

## 5. Which model fits which card

A model needs about 2 GB of card memory per billion parameters in float16, 1 GB in 8 bits and a little more than 0.5 GB in 4 bits, plus 1 to 2 GB to work.

| Card | Memory | float16 | 8 bits | 4 bits |
|---|---|---|---|---|
| T4 (Colab, Kaggle) | 15 GB | up to about 6 billion parameters | up to about 12 | up to about 20 |
| A100 (Colab Pro) | 40 GB | up to about 18 | larger | larger |

To shrink a model, add `params: {quantize: 4bit}` (or `8bit`) to its line in `experiment.yaml`, and install `bitsandbytes` first:

```python
%pip install bitsandbytes
```
