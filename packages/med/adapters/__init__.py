"""Register the medical models that Core cannot load by itself. Importing this module is enough."""
from vlmshift.extensions import register_adapter

from .chexzero import CheXzeroModel
from .open_clip_model import OpenClipModel

register_adapter("open_clip", OpenClipModel)
register_adapter("chexzero", CheXzeroModel)
