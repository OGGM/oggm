
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
    """A download failed for a transient reason after every retry.

    Parameters
    ----------
    url : str
        The URL that failed.
    attempts : int
        The number of attempts made. Zero means the host failed recently
        and the download was skipped.
    last_error : str or Exception
        The last error. An exception is stored as its ``repr``.

    Attributes
    ----------
    url : str
    attempts : int
    last_error : str
        The ``repr`` of the last exception, or a message. It is never the
        exception itself, so the error pickles across pool workers.
    """

    def __init__(self, url, attempts, last_error):
        """Initialize the DownloadOutageError.

        Args must match the signature, or unpickling in the main
        process fails if a pool worker raises this error, e.g. if a str
        is passed through, a round trip won't repr it again.
        """

        if not isinstance(last_error, str):
            last_error = repr(last_error)
        super().__init__(url, attempts, last_error)
        self.url, self.attempts, self.last_error = self.args

    def __str__(self):
        return (
            f"Download of {self.url} failed after {self.attempts} "
            f"attempts: {self.last_error}"
        )
