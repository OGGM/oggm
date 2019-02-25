
class InvalidParamsError(ValueError):
    pass


class MassBalanceCalibrationError(RuntimeError):
    pass


class InvalidDEMError(RuntimeError):
    pass


class InvalidGeometryError(RuntimeError):
    pass


class GeometryError(RuntimeError):
    pass


class NoInternetException(Exception):
    pass


class DownloadVerificationFailedException(Exception):
    def __init__(self, msg=None, path=None):
        self.msg = msg
        self.path = path


class HttpDownloadError(Exception):
    def __init__(self, code):
        self.code = code


class HttpContentTooShortError(Exception):
    pass
