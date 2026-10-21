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
    """A download ended before its Content-Length was reached.

    Attributes
    ----------
    url : str
        The URL that was being downloaded.
    """

    def __init__(self, url: str):
        self.url = url


class DownloadOutageError(RuntimeError):
    """A download failed for a transient reason after every retry.

    Attributes
    ----------
    url : str
        The URL that failed.
    attempts : int
        The number of attempts made. Zero means the host failed recently
        and the download was skipped.
    last_error : str or Exception
        The last error. An exception is stored as its ``repr``.
    """

    def __init__(
        self, url: str, attempts: int, last_error: str | BaseException | None
    ):
        """Initialise the DownloadOutageError.

        ``args`` must match the signature, or unpickling in the main
        process fails when a pool worker raises this error.
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
