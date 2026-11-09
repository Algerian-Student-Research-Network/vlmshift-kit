"""CheXzero (Tiu et al., 2022): a CLIP ViT-B/32 trained on chest X-rays of MIMIC-CXR and their reports.

The official weights are files of the OpenAI clip package, such as best_64_5e-05_original_22000_0.864.pt,
in the Google Drive folder linked from github.com/rajpurkarlab/CheXzero. open_clip reads them as they
are: in experiment.yaml, models.id is the path of one such file. The images are normalized as
CheXzero was trained (mean 101.48761 and deviation 83.43944 on the 0-255 scale), and padded to a
square instead of cut. The scoring is that of Core: cosine similarity with each sentence.
"""
from .open_clip_model import OpenClipModel

ARCHITECTURE = "ViT-B-32-quickgelu"  # the OpenAI CLIP that CheXzero starts from
MEAN, STD = 101.48761 / 255, 83.43944 / 255


class CheXzeroModel(OpenClipModel):
    def build(self, open_clip):
        model, _, preprocess = open_clip.create_model_and_transforms(
            ARCHITECTURE, pretrained=self.id, image_mean=(MEAN,) * 3, image_std=(STD,) * 3,
            image_resize_mode="longest",
        )
        return model, preprocess, open_clip.get_tokenizer(ARCHITECTURE)
