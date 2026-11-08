from urllib.parse import parse_qsl, urlencode


class RestorePath:
    """Vercel rewrites every URL to /api/index?__path=<original path>; put the original path back."""

    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        pairs = parse_qsl(environ.get("QUERY_STRING", ""), keep_blank_values=True)
        original = next((v for k, v in pairs if k == "__path"), None)
        if original is not None:
            environ["PATH_INFO"] = "/" + original.lstrip("/")
            environ["QUERY_STRING"] = urlencode([(k, v) for k, v in pairs if k != "__path"])
        return self.wsgi_app(environ, start_response)
