"""Generative models that Hugging Face transformers runs by itself (image-text-to-text).

Lighter than VLMEvalKit: nothing to install on Google Colab or Kaggle, where torch and
transformers already are. The libraries are imported only at load time.
"""
from ..base import Model, ModelError, guard
from .contrastive import open_rgb

QUANTIZE = ("8bit", "4bit")
DTYPES = ("float16", "bfloat16", "float32")


def wrap(processor, prompt: str) -> str:
    """The question as the model expects it: in its chat template if it has one (LLaVA, Qwen,
    SmolVLM), as it is otherwise (InstructBLIP, which was trained on bare instructions)."""
    if not getattr(processor, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def new_tokens(output, prompt_ids):
    """Some models give back the prompt before the answer, others only the answer."""
    length = prompt_ids.shape[1]
    if output.shape[1] >= length and bool((output[:, :length] == prompt_ids).all()):
        return output[:, length:]
    return output


class HuggingFaceModel(Model):
    kind = "generative"
    batch_size = 1  # one image at a time: each answer goes into the cache right away

    def load(self):
        quantize, dtype = self.params.get("quantize"), self.params.get("dtype")
        if quantize not in (None, *QUANTIZE):
            raise ModelError(f"params.quantize must be one of {QUANTIZE}, not {quantize!r}")
        if dtype not in (None, *DTYPES):
            raise ModelError(f"params.dtype must be one of {DTYPES}, not {dtype!r}")
        with guard():
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.torch = torch
            gpu = torch.cuda.is_available()
            if quantize and not gpu:
                raise ModelError(f"params.quantize: {quantize} needs an NVIDIA card")
            # float16 on a card (a T4 of Colab has no fast bfloat16), float32 on a processor.
            self.dtype = getattr(torch, dtype or ("float16" if gpu else "float32"))
            options = {"revision": self.revision, "dtype": self.dtype}
            if gpu:
                options["device_map"] = "auto"  # spread over several cards if one is too small
            if quantize:
                from transformers import BitsAndBytesConfig

                options["quantization_config"] = (
                    BitsAndBytesConfig(load_in_8bit=True) if quantize == "8bit"
                    else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=self.dtype)
                )
                self.notes = [f"weights quantized to {quantize}: answers can differ a little from the full model"]
            self.model = AutoModelForImageTextToText.from_pretrained(self.id, **options).eval()
            # params.processor: settings of the processor, such as max_pixels for Qwen2.5-VL and the models built on it.
            self.processor = AutoProcessor.from_pretrained(self.id, revision=self.revision, **self.params.get("processor", {}))

    def ask(self, image: str, prompt: str, params: dict) -> str:
        torch = self.torch
        inputs = self.processor(images=[open_rgb(image)], text=wrap(self.processor, prompt), return_tensors="pt")
        inputs = inputs.to(self.model.device, self.dtype)  # only the pixels change type
        temperature = params.get("temperature", 0)
        sampling = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=params.get("max_new_tokens", 16), **sampling)
        answer = new_tokens(output, inputs["input_ids"])
        return self.processor.batch_decode(answer, skip_special_tokens=True)[0].strip()

    def generate(self, images, prompts, params):
        with guard():
            return [self.ask(image, prompt, params) for image, prompt in zip(images, prompts)]
