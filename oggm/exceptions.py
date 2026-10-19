
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
    def __init__(self, url):
        self.url = url


class DownloadOutageError(RuntimeError):
    """A download failed for a transient reason after every retry."""

    def __init__(self, url, attempts, last_error):
        # args must match the signature, or unpickling in the main
        # process fails when a pool worker raises this
        super().__init__(url, attempts, repr(last_error))
        self.url, self.attempts, self.last_error = self.args

    def __str__(self):
        return (f"Download of {self.url} failed after {self.attempts} "
                f"attempts: {self.last_error}")
