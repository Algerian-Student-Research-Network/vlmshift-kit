"""Contrastive models published for the open_clip library, such as BiomedCLIP.

Hugging Face transformers cannot load them: open_clip gives the model, its image preprocessing
and its tokenizer. The scoring is that of the contrastive adapter of Core: cosine similarity.
In experiment.yaml, models.id is the open_clip name, for example
hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.
"""
from vlmshift.inference.adapters.contrastive import ContrastiveModel, open_rgb
from vlmshift.inference.base import ModelError, guard


class OpenClipModel(ContrastiveModel):
    def build(self, open_clip):
        """The model, its image preprocessing and its tokenizer."""
        model, preprocess = open_clip.create_model_from_pretrained(self.id)
        return model, preprocess, open_clip.get_tokenizer(self.id)

    def load(self):
        if self.revision:
            raise ModelError("revision: open_clip cannot choose a version. Give the path of a local copy as id")
        with guard():
            import open_clip
            import torch

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model, self.preprocess, self.tokenizer = self.build(open_clip)
            self.model = model.eval().to(self.device)

    def _image_vectors(self, paths):
        pixels = self.torch.stack([self.preprocess(open_rgb(path)) for path in paths]).to(self.device)
        with self.torch.no_grad():
            return self._unit_vectors(self.model.encode_image(pixels))

    def _text_vectors(self, texts):
        tokens = self.tokenizer(texts).to(self.device)
        with self.torch.no_grad():
            return self._unit_vectors(self.model.encode_text(tokens))
