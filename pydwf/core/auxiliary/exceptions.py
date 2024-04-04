'''The |pydwf.core.auxiliary.exceptions| module provides the |PyDwfError| and |DwfLibraryError|
exception classes.'''

from typing import Optional

from pydwf.core.auxiliary.enum_types import DwfErrorCode


class PyDwfError(Exception):
    '''A |PyDwfError| exception represents any error in |pydwf| (caused by the underlying C API or otherwise).

    It is a trivial (empty) specialization of the built-in |Exception| class.
    '''


class DwfLibraryError(PyDwfError):
    '''A |DwfLibraryError| exception represents an error reported by one of the DWF C library functions.

    This class derives from |PyDwfError|, making it easier for scripts to catch any exception originating in |pydwf|.

    The following attributes are provided:

    Attributes:

        code (Optional[DwfErrorCode]):
            DWF error code as reported by the C library, if available.

        msg (Optional[str]):
            DWF error message as reported by the C library, if available.
            It may contain multiple single-line messages, separated by a newline character.
    '''
    def __init__(self, code: Optional[DwfErrorCode], msg: Optional[str]) -> None:
        super().__init__(self)
        self.code = code
        self.msg = msg

    def __str__(self) -> str:
        if self.code is None:
            error_string = 'DWF Library Error (unspecified)'
        else:
            error_string = f'DWF Library Error {self.code.name!r} ({self.code.value})'

        if self.msg is not None:
            error_string = f'{error_string}: {self.msg.strip()!r}'

        return error_string
