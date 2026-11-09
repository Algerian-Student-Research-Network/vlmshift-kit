"""Conformity test (section 7.3): the same test for each adapter."""
import sys
import types
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import FIXTURES
from vlmshift.extensions import get_adapter, register_adapter
from vlmshift.inference.adapters import huggingface
from vlmshift.inference.adapters.contrastive import open_rgb
from vlmshift.inference.base import ModelError

IMAGES = [str(FIXTURES / "toy" / "images" / f"img{n}.png") for n in (1, 2, 3)]
BROKEN_IMAGE = "absent/missing_image.png"


def check_generative(model):
    outputs = model.generate(IMAGES, ["Is there a circle?"] * 3, {})
    assert len(outputs) == 3 and all(isinstance(output, str) for output in outputs)
    with pytest.raises(ModelError):
        model.generate([BROKEN_IMAGE], ["?"], {})


def check_contrastive(model):
    outputs = model.score(IMAGES, ["a circle"] * 3, ["no circle"] * 3)
    assert len(outputs) == 3
    assert all(isinstance(value, float) for pair in outputs for value in pair)
    with pytest.raises(ModelError):
        model.score([BROKEN_IMAGE], ["a"], ["b"])


def check_embed(model):
    vectors = model.embed(IMAGES)
    assert len(vectors) == 3 and all(isinstance(value, float) for value in vectors[0])
    with pytest.raises(ModelError):
        model.embed([BROKEN_IMAGE])


def conforms(adapter: str, id: str = "x"):
    model = get_adapter(adapter)("m", id, None, {})
    model.load()
    assert model.info()["name"] == "m"
    (check_generative if model.kind == "generative" else check_contrastive)(model)
    if model.supports_embed:
        check_embed(model)
    model.unload()


def test_fake_generative():
    conforms("fake_generative")


def test_fake_contrastive():
    conforms("fake_contrastive")


class Stub:
    """Imitates a VLMEvalKit model: a two-part message, and an error if the image is missing."""

    def __init__(self, **settings):
        self.settings = settings

    def generate(self, message):
        assert [part["type"] for part in message] == ["image", "text"]
        Path(message[0]["value"]).read_bytes()
        return "yes"


class Declared(Stub):
    """Declares its settings and refuses the others."""

    def __init__(self, temperature=1.0, model_path=None):
        super().__init__(temperature=temperature, model_path=model_path)


class Api(Stub):
    """Like a VLMEvalKit API: any undeclared setting would go to the server."""

    is_api = True
    answer = "yes"

    def __init__(self, temperature=1.0, max_tokens=2048, **others):
        super().__init__(temperature=temperature, max_tokens=max_tokens, **others)

    def generate(self, message):
        return self.answer


@pytest.fixture
def catalog(monkeypatch):
    """Install a fake VLMEvalKit whose catalog is this dictionary."""
    config = types.ModuleType("vlmeval.config")
    config.supported_VLM = {}
    monkeypatch.setitem(sys.modules, "vlmeval", types.ModuleType("vlmeval"))
    monkeypatch.setitem(sys.modules, "vlmeval.config", config)
    return config.supported_VLM


def loaded(id, revision=None, **params):
    model = get_adapter("vlmevalkit")("m", id, revision, params)
    model.load()
    return model


def test_vlmevalkit_adapter_with_a_stub_library(catalog):
    """VLMEvalKit is not installed here: we check how it is called."""
    catalog["x"] = Stub
    conforms("vlmevalkit")


def test_vlmevalkit_works_one_image_at_a_time():
    assert get_adapter("vlmevalkit")("m", "x", None, {}).batch_size == 1


def test_vlmevalkit_unknown_name_says_so(catalog):
    with pytest.raises(ModelError, match="catalog"):
        loaded("absent")


def test_vlmevalkit_declared_settings_are_applied(catalog):
    catalog["x"] = partial(Declared, model_path="org/name")
    model = loaded("x", temperature=0.5)
    assert model.model.settings == {"temperature": 0.5, "model_path": "org/name"}


def test_vlmevalkit_refused_settings_are_dropped_and_noted(catalog):
    """Before, these settings prevented the model from loading."""
    catalog["x"] = partial(Declared, model_path="org/name")
    assert loaded("x").notes == ["max_new_tokens, do_sample: refused by this model, so not applied"]


def test_vlmevalkit_undeclared_settings_are_sent_and_noted(catalog):
    catalog["x"] = Stub
    model = loaded("x", temperature=0.5, max_new_tokens=9)
    assert model.model.settings == {"temperature": 0.5, "max_new_tokens": 9, "do_sample": True}
    assert model.notes == ["temperature, max_new_tokens, do_sample: passed on without guarantee, this model does not declare it"]


def test_vlmevalkit_nothing_undeclared_reaches_an_api(catalog):
    catalog["x"] = partial(Api, model="gpt")
    model = loaded("x", max_new_tokens=9)
    assert model.model.settings == {"model": "gpt", "temperature": 0, "max_tokens": 9}  # max_tokens: the name used by APIs
    assert model.notes == ["do_sample: refused by this model, so not applied"]


def test_vlmevalkit_failure_sentence_is_a_model_error(catalog, monkeypatch):
    """A failing API does not raise an error: it returns a sentence, which is not an answer."""
    catalog["x"] = Api
    monkeypatch.setattr(Api, "answer", "Failed to obtain answer via API. Timeout")
    with pytest.raises(ModelError, match="Failed to obtain"):
        loaded("x").generate(IMAGES[:1], ["?"], {})


def test_vlmevalkit_answer_with_extra_records_gives_the_text(catalog, monkeypatch):
    catalog["x"] = Api
    monkeypatch.setattr(Api, "answer", {"prediction": "yes", "extra_records": {"tokens": 3}})
    assert loaded("x").generate(IMAGES[:1], ["?"], {}) == ["yes"]


def test_vlmevalkit_revision_loads_those_exact_weights(catalog, monkeypatch):
    calls = []
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda *args, **options: calls.append((args, options)) or "/weights/abc"
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    catalog["x"] = partial(Declared, model_path="org/name")
    assert loaded("x", revision="abc").model.settings["model_path"] == "/weights/abc"
    assert calls == [(("org/name",), {"revision": "abc", "ignore_patterns": ["onnx/*"]})]


def test_vlmevalkit_revision_impossible_to_apply_is_an_error(catalog):
    """A requested version that is not applied would give a falsely reproducible result."""
    catalog["x"] = partial(Api, model="gpt")
    with pytest.raises(ModelError, match="revision"):
        loaded("x", revision="abc")


def test_vlmevalkit_load_failure_is_a_model_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "vlmeval", None)  # library absent
    monkeypatch.setitem(sys.modules, "vlmeval.config", None)
    with pytest.raises(ModelError):
        get_adapter("vlmevalkit")("m", "x", None, {}).load()


def test_contrastive_adapter_on_a_tiny_local_clip(tmp_path):
    """A tiny random CLIP, created on the spot: the real code, without downloading."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    vocabulary = ["<|startoftext|>", "<|endoftext|>", "a</w>", "circle</w>", "no</w>"]
    (tmp_path / "vocab.json").write_text(
        "{" + ",".join(f'"{word}":{i}' for i, word in enumerate(vocabulary)) + "}", encoding="utf-8"
    )
    (tmp_path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = transformers.CLIPTokenizer(str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"))
    tokenizer.model_max_length = 8
    images = transformers.CLIPImageProcessor(size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16})
    transformers.CLIPProcessor(image_processor=images, tokenizer=tokenizer).save_pretrained(tmp_path)
    small = dict(hidden_size=8, intermediate_size=8, num_hidden_layers=1, num_attention_heads=2)
    config = transformers.CLIPConfig(
        text_config={**small, "vocab_size": 5, "max_position_embeddings": 8},
        vision_config={**small, "image_size": 16, "patch_size": 8},
        projection_dim=4,
    )
    transformers.CLIPModel(config).save_pretrained(tmp_path)
    conforms("contrastive", str(tmp_path))


def test_load_failure_of_a_contrastive_model_is_a_model_error(tmp_path):
    pytest.importorskip("transformers")
    with pytest.raises(ModelError):
        get_adapter("contrastive")("m", str(tmp_path / "absent"), None, {}).load()


def test_a_package_can_register_its_adapter():
    class Mine(get_adapter("fake_generative")):
        pass

    register_adapter("mine", Mine)
    assert get_adapter("mine") is Mine


def test_a_16_bit_image_keeps_its_contrast(tmp_path):
    path = tmp_path / "xray.png"
    Image.fromarray(np.linspace(0, 4095, 256 * 16).reshape(16, 256).astype(np.uint16)).save(path)  # 12 bits, as many X-rays
    pixels = np.asarray(open_rgb(str(path)))[..., 0]
    assert pixels.min() == 0 and pixels.max() == 255
    assert (pixels == 255).mean() < 0.01  # converted directly, almost every pixel would be white
    assert 120 <= np.median(pixels) <= 135


def tiny_llava(folder: Path) -> str:
    """A tiny random LLaVA, created on the spot: the real code of transformers, without downloading."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    words = ["<unk>", "<s>", "</s>", "<image>", "USER:", "ASSISTANT:", "Is", "there", "a", "circle?"]
    core = tokenizers.Tokenizer(tokenizers.models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    core.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=core, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<unk>",
        extra_special_tokens={"image_token": "<image>"},
    )
    template = (
        "{% for m in messages %}USER: {% for c in m['content'] %}{% if c['type'] == 'image' %}<image> "
        "{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endfor %}{% if add_generation_prompt %} ASSISTANT:{% endif %}"
    )
    transformers.LlavaProcessor(
        image_processor=transformers.CLIPImageProcessor(size={"shortest_edge": 16}, crop_size={"height": 16, "width": 16}),
        tokenizer=tokenizer, patch_size=8, vision_feature_select_strategy="default", chat_template=template,
        image_token="<image>", num_additional_image_tokens=1,  # the CLS token of the vision encoder
    ).save_pretrained(folder)
    small = dict(hidden_size=8, intermediate_size=8, num_hidden_layers=1, num_attention_heads=2)
    config = transformers.LlavaConfig(
        vision_config=transformers.CLIPVisionConfig(**small, image_size=16, patch_size=8),
        text_config=transformers.LlamaConfig(**small, vocab_size=len(words), num_key_value_heads=2, max_position_embeddings=64),
        image_token_index=3, vision_feature_select_strategy="default", vision_feature_layer=-1,
    )
    transformers.LlavaForConditionalGeneration(config).save_pretrained(folder)
    return str(folder)


def test_huggingface_adapter_on_a_tiny_local_llava(tmp_path):
    conforms("huggingface", tiny_llava(tmp_path))


def test_huggingface_answer_holds_only_the_new_words(tmp_path):
    model = get_adapter("huggingface")("m", tiny_llava(tmp_path), None, {})
    model.load()
    # The question is wrapped as the model was trained to read it.
    assert huggingface.wrap(model.processor, "Is there a circle?") == "USER: <image> Is there a circle? ASSISTANT:"
    answer = model.generate(IMAGES[:1], ["Is there a circle?"], {"max_new_tokens": 3, "temperature": 0})[0]
    # The same call made by hand: LLaVA gives back the prompt, then the new words.
    inputs = model.processor(images=[open_rgb(IMAGES[0])], text="USER: <image> Is there a circle? ASSISTANT:", return_tensors="pt")
    output = model.model.generate(**inputs, max_new_tokens=3, do_sample=False)
    length = inputs["input_ids"].shape[1]
    assert output[:, :length].tolist() == inputs["input_ids"].tolist()
    assert answer == model.processor.batch_decode(output[:, length:], skip_special_tokens=True)[0].strip()


def test_huggingface_processor_settings_reach_the_processor(tmp_path):
    # params.processor goes to AutoProcessor, as max_pixels for Qwen2.5-VL in the Med experiment.
    model = get_adapter("huggingface")("m", tiny_llava(tmp_path), None, {"processor": {"size": {"shortest_edge": 32}}})
    model.load()
    assert model.processor.image_processor.size["shortest_edge"] == 32


def test_huggingface_without_chat_template_sends_the_bare_question():
    # InstructBLIP has no chat template: it was trained on bare instructions, as in the POPE paper.
    assert huggingface.wrap(types.SimpleNamespace(chat_template=None), "Is there a dog?") == "Is there a dog?"


def test_huggingface_keeps_the_answer_when_the_prompt_is_not_given_back():
    torch = pytest.importorskip("torch")
    prompt = torch.tensor([[5, 6, 7]])
    assert huggingface.new_tokens(torch.tensor([[5, 6, 7, 1, 2]]), prompt).tolist() == [[1, 2]]
    assert huggingface.new_tokens(torch.tensor([[1, 2]]), prompt).tolist() == [[1, 2]]
    assert huggingface.new_tokens(torch.tensor([[9, 6, 7, 1]]), prompt).tolist() == [[9, 6, 7, 1]]


@pytest.mark.parametrize("params, message", [
    ({"quantize": "3bit"}, "params.quantize must be one of"),
    ({"dtype": "float8"}, "params.dtype must be one of"),
])
def test_huggingface_wrong_settings_are_refused(params, message):
    with pytest.raises(ModelError, match=message):
        get_adapter("huggingface")("m", "x", None, params).load()


def test_huggingface_quantizing_without_a_card_is_refused(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ModelError, match="needs an NVIDIA card"):
        get_adapter("huggingface")("m", str(tmp_path), None, {"quantize": "4bit"}).load()


def word_tokenizer(words: list[str], **special):
    """A tokenizer that knows a few words, built without downloading."""
    transformers, tokenizers = pytest.importorskip("transformers"), pytest.importorskip("tokenizers")
    core = tokenizers.Tokenizer(tokenizers.models.WordLevel({w: i for i, w in enumerate(words)}, unk_token=words[0]))
    core.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=core, **special)


def test_huggingface_instructblip_gets_the_bare_question(tmp_path):
    """InstructBLIP, the model of the POPE paper, through the same adapter: a tiny random one."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("torch")
    words = ["<unk>", "<s>", "</s>", "<image>", "Is", "there", "a", "dog", "in", "the", "image?"]
    transformers.InstructBlipProcessor(
        image_processor=transformers.BlipImageProcessor(size={"height": 16, "width": 16}),
        tokenizer=word_tokenizer(words, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<unk>"),
        qformer_tokenizer=word_tokenizer(["[UNK]", "[CLS]", "[SEP]", "[PAD]", "dog"], unk_token="[UNK]",
                                         cls_token="[CLS]", sep_token="[SEP]", pad_token="[PAD]"),
        num_query_tokens=2,
    ).save_pretrained(tmp_path)
    small = dict(hidden_size=8, intermediate_size=8, num_hidden_layers=1, num_attention_heads=2)
    transformers.InstructBlipForConditionalGeneration(transformers.InstructBlipConfig(
        vision_config={**small, "image_size": 16, "patch_size": 8},
        qformer_config={**small, "vocab_size": 5, "encoder_hidden_size": 8, "max_position_embeddings": 32},
        text_config=transformers.LlamaConfig(**small, vocab_size=len(words), num_key_value_heads=2, max_position_embeddings=64).to_dict(),
        num_query_tokens=2, image_token_index=3,
    )).save_pretrained(tmp_path)
    model = get_adapter("huggingface")("m", str(tmp_path), None, {})
    model.load()
    assert huggingface.wrap(model.processor, "Is there a dog in the image?") == "Is there a dog in the image?"
    answer = model.generate(IMAGES[:1], ["Is there a dog in the image?"], {"max_new_tokens": 2})[0]
    inputs = model.processor(images=[open_rgb(IMAGES[0])], text="Is there a dog in the image?", return_tensors="pt")
    output = model.model.generate(**inputs, max_new_tokens=2, do_sample=False)
    new = huggingface.new_tokens(output, inputs["input_ids"])
    assert new.shape[1] <= 2  # whatever this version of transformers gives back, only the new words remain
    assert answer == model.processor.batch_decode(new, skip_special_tokens=True)[0].strip()


def test_huggingface_a_retried_load_gives_its_note_once(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from vlmshift.inference.runner import attempt

    def absent(*args, **kwargs):
        raise OSError("absent/absent is not a model")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)  # the quantize branch needs a card
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained", absent)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", dict)  # the real one takes 20 s to import
    model = get_adapter("huggingface")("m", "absent/absent", None, {"quantize": "8bit"})
    with pytest.raises(ModelError, match="not a model"):
        attempt(model.load, 2)
    assert len(model.notes) == 1
