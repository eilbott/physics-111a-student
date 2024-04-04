'''This module provides the :py:class:`typespec_ctypes` class.

It is used to map the C types found in the 'dwf.h' header file to specific *ctypes* types.
'''

import ctypes


class typespec_ctypes:
    '''Map the type specifications from :py:mod:`pydwf.core.auxiliary.dwf_function_signatures` to *ctypes* types.'''

    # pylint: disable=too-few-public-methods

    # The basic C types.
    #
    # Note: the use of type 'bool' in dwf.h is probably not intentional;
    #  it is not a C type (unless stdbool.h bas been included).

    c_bool                                = ctypes.c_bool

    c_char                                = ctypes.c_char
    c_char_ptr                            = ctypes.POINTER(c_char)
    c_const_char_ptr                      = ctypes.c_char_p

    c_char_array_16                       = c_char * 16
    c_char_array_32                       = c_char * 32
    c_char_array_512                      = c_char * 512

    c_short                               = ctypes.c_short
    c_short_ptr                           = ctypes.POINTER(c_short)

    c_int                                 = ctypes.c_int
    c_int_ptr                             = ctypes.POINTER(c_int)

    c_unsigned_char                       = ctypes.c_ubyte
    c_unsigned_char_ptr                   = ctypes.POINTER(c_unsigned_char)

    c_unsigned_short                      = ctypes.c_ushort
    c_unsigned_short_ptr                  = ctypes.POINTER(c_unsigned_short)

    c_unsigned_int                        = ctypes.c_uint
    c_unsigned_int_ptr                    = ctypes.POINTER(c_unsigned_int)

    c_unsigned_long_long                  = ctypes.c_ulonglong
    c_unsigned_long_long_ptr              = ctypes.POINTER(c_unsigned_long_long)

    c_double                              = ctypes.c_double
    c_double_ptr                          = ctypes.POINTER(c_double)
    c_double_array_32                     = c_double * 32

    c_const_double                        = c_double
    c_const_double_ptr                    = ctypes.POINTER(c_const_double)

    c_void_ptr                            = ctypes.c_void_p

    # the HDWF type represents an open device handle.

    HDWF                                  = c_int
    HDWF_ptr                              = ctypes.POINTER(HDWF)

    # The enumeration types that occur in dw.h function prototypes follow.
    # Note: some of them were renamed from their C counterparts.

    DwfErrorCode                          = c_int
    DwfErrorCode_ptr                      = ctypes.POINTER(DwfErrorCode)

    DwfEnumFilter                         = c_int

    DwfEnumConfigInfo                     = c_int

    DwfDeviceID                           = c_int
    DwfDeviceID_ptr                       = ctypes.POINTER(DwfDeviceID)

    DwfDeviceVersion                      = c_int
    DwfDeviceVersion_ptr                  = ctypes.POINTER(DwfDeviceVersion)

    DwfDeviceParameter                    = c_int

    DwfWindow                             = c_int
    DwfWindow_ptr                         = ctypes.POINTER(DwfWindow)

    DwfState                              = c_unsigned_char
    DwfState_ptr                          = ctypes.POINTER(DwfState)

    DwfTriggerSource                      = c_unsigned_char
    DwfTriggerSource_ptr                  = ctypes.POINTER(DwfTriggerSource)

    DwfTriggerSlope                       = c_int
    DwfTriggerSlope_ptr                   = ctypes.POINTER(DwfTriggerSlope)

    DwfAcquisitionMode                    = c_int
    DwfAcquisitionMode_ptr                = ctypes.POINTER(DwfAcquisitionMode)

    DwfAnalogInFilter                     = c_int
    DwfAnalogInFilter_ptr                 = ctypes.POINTER(DwfAnalogInFilter)

    DwfAnalogInTriggerType                = c_int
    DwfAnalogInTriggerType_ptr            = ctypes.POINTER(DwfAnalogInTriggerType)

    DwfAnalogInTriggerLengthCondition     = c_int
    DwfAnalogInTriggerLengthCondition_ptr = ctypes.POINTER(DwfAnalogInTriggerLengthCondition)

    DwfAnalogCoupling                     = c_int
    DwfAnalogCoupling_ptr                 = ctypes.POINTER(DwfAnalogCoupling)

    DwfAnalogOutFunction                  = c_unsigned_char
    DwfAnalogOutFunction_ptr              = ctypes.POINTER(DwfAnalogOutFunction)

    DwfAnalogOutNode                      = c_int

    DwfAnalogOutMode                      = c_int
    DwfAnalogOutMode_ptr                  = ctypes.POINTER(DwfAnalogOutMode)

    DwfAnalogOutIdle                      = c_int
    DwfAnalogOutIdle_ptr                  = ctypes.POINTER(DwfAnalogOutIdle)

    DwfDigitalInClockSource               = c_int
    DwfDigitalInClockSource_ptr           = ctypes.POINTER(DwfDigitalInClockSource)

    DwfDigitalInSampleMode                = c_int
    DwfDigitalInSampleMode_ptr            = ctypes.POINTER(DwfDigitalInSampleMode)

    DwfDigitalOutOutput                   = c_int
    DwfDigitalOutOutput_ptr               = ctypes.POINTER(DwfDigitalOutOutput)

    DwfDigitalOutType                     = c_int
    DwfDigitalOutType_ptr                 = ctypes.POINTER(DwfDigitalOutType)

    DwfDigitalOutIdle                     = c_int
    DwfDigitalOutIdle_ptr                 = ctypes.POINTER(DwfDigitalOutIdle)

    DwfAnalogIO                           = c_unsigned_char
    DwfAnalogIO_ptr                       = ctypes.POINTER(DwfAnalogIO)

    DwfAnalogImpedance                    = c_int
