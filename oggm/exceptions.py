
class InvalidParamsError(ValueError):
    pass


class InvalidWorkflowError(ValueError):
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


class DownloadCredentialsMissingException(Exception):
    pass


class DownloadVerificationFailedException(Exception):
    def __init__(self, msg=None, path=None):
        self.msg = msg
        self.path = path


class HttpDownloadError(Exception):
    def __init__(self, code, url):
        self.code = code
        self.url = url


class FTPSDownloadError(Exception):
    def __init__(self, orgerr):
        self.orgerr = orgerr


class HttpContentTooShortError(Exception):
    pass
