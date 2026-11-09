"""Contrastive models (CLIP and similar) through Hugging Face.

The score of a phrase is its cosine similarity with the image.
The libraries are imported only at load time.
"""
from ...common import DEEP_MODES
from ..base import Model, guard


def open_rgb(path: str):
    """An image as the model expects it. A 16-bit image, common for X-rays, is scaled to 8 bits:
    converted directly, almost every pixel would turn white."""
    import numpy as np
    from PIL import Image

    image = Image.open(path)
    if image.mode in DEEP_MODES:
        pixels = np.asarray(image, dtype=np.float64)
        low, high = pixels.min(), pixels.max()
        scaled = (pixels - low) / (high - low) * 255 if high > low else np.zeros_like(pixels)
        image = Image.fromarray(scaled.round().astype(np.uint8))
    return image.convert("RGB")


class ContrastiveModel(Model):
    kind = "contrastive"
    supports_embed = True

    def load(self):
        with guard():
            import torch
            from transformers import AutoModel, AutoProcessor

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            weights = AutoModel.from_pretrained(self.id, revision=self.revision)
            self.model = weights.eval().to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.id, revision=self.revision)

    def _unit_vectors(self, features):
        # Depending on the transformers version, we get a tensor or an object that holds it.
        features = getattr(features, "pooler_output", features)
        return features / features.norm(dim=-1, keepdim=True)

    def _image_vectors(self, paths):
        pictures = [open_rgb(path) for path in paths]
        inputs = self.processor(images=pictures, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            return self._unit_vectors(self.model.get_image_features(**inputs))

    def _text_vectors(self, texts):
        inputs = self.processor(
            text=texts, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            return self._unit_vectors(self.model.get_text_features(**inputs))

    def score(self, images, positives, negatives):
        with guard():
            vectors = self._image_vectors(images)
            positive = (vectors * self._text_vectors(positives)).sum(dim=-1)
            negative = (vectors * self._text_vectors(negatives)).sum(dim=-1)
            return list(zip(positive.tolist(), negative.tolist()))

    def embed(self, images):
        with guard():
            return self._image_vectors(images).tolist()
