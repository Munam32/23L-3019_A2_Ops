import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onnx_infer import OnnxClassifier  # noqa: E402
from restore_path import RestorePath  # noqa: E402
from service import create_api  # noqa: E402

os.environ.setdefault("MAX_UPLOAD_MB", "4")
app = create_api(OnnxClassifier())
app.wsgi_app = RestorePath(app.wsgi_app)
