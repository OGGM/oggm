
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
    pass


class HttpDownloadError(Exception):
    def __init__(self, code):
        self.code = code


class HttpContentTooShortError(Exception):
    pass
