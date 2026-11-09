"""Generative models through VLMEvalKit. The library is imported only at load time.

VLMShift calls the catalog model directly, without going through the VLMEvalKit
pipeline: the text sent stays exactly the one prepared in step 3.
"""
from functools import partial
from inspect import Parameter, signature

from ..base import Model, ModelError, guard

FAILED = "Failed to obtain answer"  # VLMEvalKit returns this sentence instead of raising an error


def split_settings(factory, settings: dict) -> tuple[dict, list[str]]:
    """Keep the settings the model can receive. Return (settings, notes for the run log).

    Each model has its own settings: a name forced on all of them would block some,
    and a setting silently ignored would make the run log lie.
    """
    target = factory.func if isinstance(factory, partial) else factory
    declared = signature(target).parameters
    if "max_tokens" in declared and "max_new_tokens" not in declared:  # the name used by APIs
        settings = {"max_tokens" if name == "max_new_tokens" else name: value for name, value in settings.items()}
    # An API would send a setting it does not declare to the server as it is.
    open_ended = any(item.kind is Parameter.VAR_KEYWORD for item in declared.values())
    accepts_others = open_ended and not getattr(target, "is_api", False)

    kept = {name: value for name, value in settings.items() if name in declared or accepts_others}
    unchecked = [name for name in kept if name not in declared]
    dropped = [name for name in settings if name not in kept]
    notes = []
    if dropped:
        notes.append(f"{', '.join(dropped)}: refused by this model, so not applied")
    if unchecked:
        notes.append(f"{', '.join(unchecked)}: passed on without guarantee, this model does not declare it")
    return kept, notes


def pinned_weights(factory, revision: str) -> str:
    """Download the weights of this exact version: revision then has a real effect."""
    repository = getattr(factory, "keywords", {}).get("model_path")
    if not repository:
        raise ModelError("revision requested, but this model has no model_path to pin")
    from huggingface_hub import snapshot_download

    # Some repositories also hold other formats, several GB that transformers never reads.
    return snapshot_download(repository, revision=revision, ignore_patterns=["onnx/*"])


class VLMEvalKitModel(Model):
    kind = "generative"
    batch_size = 1  # VLMEvalKit handles one image at a time: each answer goes into the cache right away

    def load(self):
        with guard():
            from vlmeval.config import supported_VLM

            if self.id not in supported_VLM:
                raise ModelError(f"{self.id} is not in the VLMEvalKit catalog")
            factory = supported_VLM[self.id]
            temperature = self.params.get("temperature", 0)
            settings, self.notes = split_settings(factory, {
                "temperature": temperature,
                "max_new_tokens": self.params.get("max_new_tokens", 16),
                "do_sample": temperature > 0,
            })
            if self.revision:
                settings["model_path"] = pinned_weights(factory, self.revision)
            self.model = factory(**settings)

    def ask(self, image: str, prompt: str) -> str:
        # Explicit types: an invalid image path gives an error, not a question without an image.
        answer = self.model.generate([dict(type="image", value=image), dict(type="text", value=prompt)])
        if isinstance(answer, dict):  # some APIs add information to the answer
            answer = answer["prediction"]
        if FAILED in str(answer):
            raise ModelError(str(answer))
        return answer

    def generate(self, images, prompts, params):
        with guard():
            return [self.ask(image, prompt) for image, prompt in zip(images, prompts)]
