import io
import os

from PIL import Image
from werkzeug.utils import secure_filename

ALLOWED = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}}
ALLOWED_EXT = {e for exts in ALLOWED.values() for e in exts}
UNDECODABLE = "This file looks like an image but cannot be decoded. Please upload an intact JPG or PNG."


def read_upload(file):
    name = secure_filename(file.filename) if file and file.filename else ""
    if not name:
        raise ValueError("Choose an image file first.")
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_EXT:
        raise ValueError("Only JPG and PNG images are accepted.")
    data = file.read()
    if not data:
        raise ValueError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
            fmt = im.format
    except Exception:
        raise ValueError(UNDECODABLE) from None
    if fmt not in ALLOWED or ext not in ALLOWED[fmt]:
        raise ValueError("The file content does not match its extension.")
    return name, data
