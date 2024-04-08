#! /usr/bin/env python3

# Old F23 version, do not use!

""" Lab 10.7 Heart Rate Lock-in Modulation 
    
    Based on python waveform demo

Summary
-------

This demo generates 
1) a signal on wavegen channel 1 (analogOut)
2) records traces of various lengths on oscilloscope channels 1 and 2 (analogIn)
3) demodulates signal
4) makes plots and ffts

Description
-----------
main()
       
the openDwfDevice function tries to connect to the physical ADS. If it succeeds, it makes
an object called device which corresponds to the ADS. Basically our entire python code runs
inside of the try because we want to be handle various failures gracefully (connecting and 
interacting with hardware can definitely produce exceptions). 

Then we create the our analogIn (the oscilloscope) and analogOut (the wavegen) objects. 

Next, we configure our wavegen with the function configure_analog_output. After this, 
the code is currently setup to purposefully get stuck in a infinite while loop (this will
be commented out later when we want to start to understand the oscilloscope)


Next, take a look at configure_analog_output. All of analogOut's properties are set in this
function, so you will be making lots of changes to this function!

    If you only want to excite channel 1 and do simple excitations (i.e. not making complex 
    radio signals which won't be needed in this lab), almost all the parameters that you need
    to change are located under ``//////////Student Reconfigurable WaveGen Properties ///////////'' 
            
    The only other wavegen parameter that is set somewhere else is the wavegen excitation frequency, 
    ANALOG_OUT_FREQUENCY. This is set at the very top of \texttt{lab10/wavegen\_tutorial.py} so that
    its global and can be seen by all functions. 

    To understand what is meant by node it's best to look at the wavegen GUI in waveforms and set the
    menu to Modulation. 

    The "analog out node" can have values of either "Carrier", "AM", "FM". If you wanted an AM modulated
    signal you would first change the parameters for the carrier signal and then change the parameters for
    AM signal. However, when you just want the wavegen to output a ``simple'' wave form  you use the Carrier node. 
    
    Finally, the command analogOut.configure(CH, True) is the equivalent of the green arrow run button
    in waveforms; it's starts the operation of the wavegnen. 
    
Next, configure_analog_in configures the properties of analogIn, which is the oscilloscope object.
The final function call is to run_acquisition}. This function starts the oscilloscope, takes traces
(or records as pyDWF calls them), and makes various plots of the individual traces.

Let's briefly look inside configure_analog_in. Currently there are only 5 settings that you should really modify:
    - ANALOG\_IN\_SAMPLE\_FREQUENCY and RECORD\_LENGTH\_SAMPLE: the names are self explanatory. They are both 
    located at the top of wavegen\_tutorial.py since they are used by multiple functions. 
    
    -TRIGGER\_LEVEL and CH_TRIGGER: These are located at the top of configure_analog_in and are again
    hopefully self explanatory. \texttt{CH\_TRIGGER} determines which analogIn channel is being used to trigger.

    - channels: this list determines which oscilloscope channels are enabled and readout. 
        
Briefly, this function:   
    -resets the oscilloscope to default values
    -sets the frequency and record length (trace length)
    -chooses the data taking mode:
        - if the record length is shorter than the ADC buffer size, then the oscilloscope is automatically put into \texttt{single1} mode. In this mode, after a trigger occurs the entire ADC buffer is filled with data and then acquistion stops.  
        - if the record length is longer than the ADC buffer size, the the oscilloscope will operate in \texttt{record} mode where it will readout the cyclical buffer while data is being written to the buffer. This operation is a bit more finicky. In particular, if readout of the ADC buffer is too slow you can lose data. Also, I've noticed that the act of reading data from the ADC and sending to the laptop can cause noise on the signal generator. I'm actively playing around to improve the performance of this mode, but you will need it for this lab. 
            
    -next it enables the oscilloscope channels that you will read out
    -it then configures the oscilloscope trigger. Unfortunately, you don't have as much freedom here as I hoped. In particular, I have failed after an absurd number of hours to get ``sychronization'' working where the trigger signal that starts the wavegen is used to trigger the oscilloscope too. Implementing this will likely require some discussions with diligent. Right now we only have triggering on the scope itself working. Practically, this won't be an issue for this lab. %There is also a bug in the \texttt{trigger\_position} definition that I need to track down (it's in the python wrapper for the C++ code). So we're just keeping this to zero. 
    
the next step in main() is a function call to run\_acquistion which     
    starts oscilloscope acquisition 
    plots traces as they come in one-by-one using \texttt{plot\_record} and \texttt{plot\_fft}
    stops acquisition after \texttt{NUMBER\_RECORDS} traces have been read."""

import time
import numpy as np
import matplotlib.pyplot as plt

from pydwf import (
    DwfLibrary,
    DwfEnumConfigInfo,
    DwfAnalogOutNode,
    DwfAnalogOutFunction,
    DwfAcquisitionMode,
    DwfTriggerSource,
    DwfAnalogInTriggerType,
    DwfTriggerSlope,
    DwfState,
    DwfAnalogInFilter,
    PyDwfError,
)
from pydwf.utilities import openDwfDevice

#------------ Global Settings--------------
lgc_test=True #just plots the trace plot vs time
lgc_demodulate = False #Calls the demodulation function and plots 
lgc_lockIn = False #Calls the Lock-In function and plots


# ----- WaveGen Time Properties (analogOut) ----- 
# TODO 10.2.e: edit the frequency so that it has the correct values
#Student Version Initial Settings
ANALOG_OUT_FREQUENCY = 100 #[Hz]
#10.2.e Solution

#Other useful settings
ANALOG_OUT_FREQUENCY = 1e3 #[Hz]

""" 
In theory the wavegen trigger should be absolutely synchronized to the oscilloscope trigger
# but I've noticed some potential issues, so I think you should consider calculating the 
# demodulation phase shift in demodulation/lock-in routine. You can then hardcode it after you've run a bit.
"""
PHASE_DEMOD = None # it's calculated in demodulate/lock-in routine.
#PHASE_DEMOD = -83.6*np.pi/180 #[radians] This is the value that I get in single mode ... beware it may change!

#------------ Oscilloscope / ADC Parameters (analogIn Object) -----
#TODO Students should modify Oscilloscope Parameters in 10.3.g

#You can always choose 2 out of 3 of RECORD_LENGTH_SAMPLES, Record_Length_Time, 
# and ANALOG_IN_SAMPLE_FREQUENCY.

# Option 1: For the Lock-in let's solve for the ANALOG_IN_SAMPLE_FREQUENCY in terms of the other 2
#RECORD_LENGTH_SAMPLES= 16384
#Record_Length_Time= 50e-3 #[s]
#ANALOG_IN_SAMPLE_FREQUENCY = 1/Record_Length_Time *RECORD_LENGTH_SAMPLES #[Hz]

# Option 1b: If you just want to see periodic number of traces
# Initial Settings for student version
RECORD_LENGTH_SAMPLES= 16384
Record_Length_Time= 20/ANALOG_OUT_FREQUENCY #[s]
ANALOG_IN_SAMPLE_FREQUENCY = 1/Record_Length_Time *RECORD_LENGTH_SAMPLES #[Hz]

#Option 2: For the demodulator, I suggests that you choose ANALOG_IN_SAMPLE_FREQUENCY, Record_Length_Time 
#ANALOG_IN_SAMPLE_FREQUENCY = 40*ANALOG_OUT_FREQUENCY #[Hz]
#Record_Length_Time = 5 #[s]
#RECORD_LENGTH_SAMPLES = round(Record_Length_Time*ANALOG_IN_SAMPLE_FREQUENCY)
#We must recalculate Record_Length_Time due to the fact that RECORD_LENGTH_SAMPLES is integer
#Record_Length_Time =  RECORD_LENGTH_SAMPLES/ANALOG_IN_SAMPLE_FREQUENCY#[s]

# Total number of records (traces) that are taken before the code ends
#NUMBER_RECORDS = 3
NUMBER_RECORDS = round(10/Record_Length_Time) #let's run for 20s before stopping program 

#--------------- WaveGen Config Function (analogOut) -----------------------------------
def configure_analog_out(analogOut, ANALOG_OUT_FREQUENCY):
    """ This function configues the analogOut Object which is the virtual WaveGen Object. Nearly 
     All the properties have a 1-to-1 correspondence with the GUI ... the one exception being I can't 
     figure out how to synchronize the wavegen and oscilloscope triggers :(

    # Inputs:
    #    1) analogOut: virtual object that controls the ADS wavegen 
    #    2) analog_out_frequency: frequency of the wavegen 
    #        note: right now there is no functionality for different """

    # //////////Student Reconfigurable WaveGen Properties ///////////
    # TODO: 10.2.e: Modify these parameters as appropriate
     
    #---- 10.2.e initial Student Version Settings -----
    ANALOG_OUT_AMPLITUDE = 1.1 #[V] wavegen amplitude
    ANALOG_OUT_OFFSET = 1.0  #[V] wavegen offset
    nodeFunctionSymmetry=50
    nodeFunctionShape = DwfAnalogOutFunction.Triangle
    """ Other Options for DwfAnalogOutFunction
      #1) SQUARE
      #2) DC
      #3) Sine
      # ... see pyDWF documentation for more options or just probe the enum directly"""
    
    #---- 10.2.e Solution -----
    #?
    #---- Rough optimum settings for heart rate for my configuration (yours is probably different)-----
    #ANALOG_OUT_AMPLITUDE = 2.3 #[V]
    #ANALOG_OUT_OFFSET = .3 #[V]
    #nodeFunctionShape = DwfAnalogOutFunction.Sine
    #nodeFunctionSymmetry=50

    #////////////////////////////////////////////////////////////

    analogOut.reset(-1)  # Reset both channels to default values

    #CH is the channel # variable
    CH = 0  #channel 1
    #CH = 1 #channel 2

    """node: each WaveGen channel has 3 `nodes' corresponding to the parameter columns of the 
    # modulation tab in waveforms wavegen GUI
    #    1) Carrier: Use this for node for simple waveforms as well 
    #    2) AM:
    #    3) FM:"""   
    node = DwfAnalogOutNode.Carrier
    #node =  DwfAnalogOutNode.AM

    """
    MP: pydwf says there is no good documenation as to what this is doing
    MP: I think this is just the enable true/false button on the waveforms 
    wavegen/modulation tab"""
    
    analogOut.nodeEnableSet(CH, node, True)

    analogOut.nodeFunctionSet(CH, node, nodeFunctionShape)
    analogOut.nodeFrequencySet(CH, node, ANALOG_OUT_FREQUENCY)
    analogOut.nodeAmplitudeSet(CH, node,  ANALOG_OUT_AMPLITUDE)
    analogOut.nodeOffsetSet(CH, node, ANALOG_OUT_OFFSET)
    analogOut.nodeSymmetrySet(CH, node, nodeFunctionSymmetry) 

    # Start the wavegen operating
    # This is the equivalent of the run (green arrow) on the ADS
    analogOut.configure(CH, True)

#--------------- Configure Oscilloscope & Trigger Function (analogIn) -----------------------------------
def configure_analog_in(analogIn, trigger_flag):
    #Channel(s) that we will configure
    CH1 = 0 # index 0 is for Channel 1 
    CH2 = 1 #index 1 is for Channel 2

    #////////// analogIn Parameters to Modify /////////////////
    # TODO Students feel free to modify as appropriate (10.3.f and later)

    """This is the channel range
       - If signal is inside the range <+/- 2.5V put  "5"V here
       - If signal is >+/- 2.5V put "24"V here
     This sets an attenuator right before the ADC of the ADS oscilloscope"""
    #CHANNEL_RANGE=[5,5]#[V]
    CHANNEL_RANGE=[24,24]#V
    #CHANNEL_RANGE=[24,5]#V

    #--- initial student setting is DetectorAnalogIn
    TRIGGER_SOURCE = DwfTriggerSource.DetectorAnalogIn #trigger on scope channel
    #TRIGGER_SOURCE = DwfTriggerSource.AnalogOut1 #trigger on wavegen 
    
    #Trigger level only used when triggering on DetectorAnalogIn (scope)
    TRIGGER_LEVEL = 0 #[V]
    # Which channel you trigger on (only used with analogIn trigger)
    CH_TRIGGER = CH1

    #Modify the channels list if you want to readout more than just channel 1   
    channels = (CH1,CH2)
    
    #/////////////////////////////////////////////////


    #Let's just make certain that analogIn is in its default settings
    analogIn.reset()

    analogIn.frequencySet(ANALOG_IN_SAMPLE_FREQUENCY)  #digitization rate of recorded trace
    analogIn.recordLengthSet(Record_Length_Time)
     

    #------ Choose data taking mode -------
    #Now let's set the aquisition mode
    
    #print("analogIn.mode: {}".format(analogIn.acquisitionModeGet()))
    if RECORD_LENGTH_SAMPLES <= analogIn.bufferSizeGet():
        """ 
        For record lengths that are shorter than the buffer, there is no reason to go continuous mode. 

        Both Single mode and Single1 mode:
             1)search for a trigger 
             2) post trigger they fill their buffer completely 
             3) they stop acquiring data

        In dwf 3.19.5 there were differences between Single1 and Single, however in 3.21.3 I haven't 
        found any differences between the 2 modes.
        """
        
        #analogIn.acquisitionModeSet(DwfAcquisitionMode.Single1)
        analogIn.acquisitionModeSet(DwfAcquisitionMode.Single)

        # In record mode, trigger_position =0 means the trigger is at the center of the record. 
        # Let's make it so that the trigger is always at the beginning of the record
        trigger_position= -Record_Length_Time/2#[s]
        #trigger_position= 0

        #TODO decrease buffer size to exactly the trace length size to increase speed?
        print("buffer size before change: {} ".format(analogIn.bufferSizeGet()))
        analogIn.bufferSizeSet(RECORD_LENGTH_SAMPLES)
        print("buffer size after change: {}".format(analogIn.bufferSizeGet()))

    else:
        """ 
        For records that are longer than the buffer size use record mode. This mode allows you to read data from the cyclical
        ADC buffer while writing is occuring. Of course, if you can't remove as fast as you create, you end up with lost data.
        You must check for this.
        """ 
        analogIn.acquisitionModeSet(DwfAcquisitionMode.Record)

        # In record mode, trigger_position =0 means the trigger is at the beginning of the record 
        trigger_position= 0
    print("Acquistion Mode: {}".format(analogIn.acquisitionModeGet()))    

      
    for channel_index in channels:
        analogIn.channelEnableSet(channel_index, True)  # Enables the scope channel

        #The actual sample frequency is much higher that wanted sample frequency here we choose to average
        # all the measurements below the wanted sample frequency 
        analogIn.channelFilterSet(channel_index, DwfAnalogInFilter.Average)

        #Voltage input range of scope ADC 
        analogIn.channelRangeSet(channel_index, CHANNEL_RANGE[channel_index])  # Range of scope
        

    #---------- Signal Generator Trigger Setup -----------
    # MP: AnalogOut1 isn't sync'ing the wavegen and the oscilloscope. I can't figure out how to enable "synchronize"
    if trigger_flag:
        
        #--- Trigger Source ---
        analogIn.triggerSourceSet(TRIGGER_SOURCE) 
        analogIn.triggerChannelSet(CH_TRIGGER)
        
        #------ trigger position -------
        """"
        from pyDWF documentation: 
            Set the AnalogIn instrument trigger position, in seconds.The meaning of the trigger 
            position depends on the currently selected acquisition mode:
        
            In Record acquisition mode, the trigger position is the time of the first valid sample acquired relative
            to the position of the trigger event. Negative values indicates times before the trigger time. To place
            the trigger in the middle of the recording, this value should be set to -0.5 times the duration of the recording.

            In Single acquisition mode, the trigger position is the trigger event time relative to the center of the
            acquisition window.To place the trigger in the middle of the acquisition buffer, the value should be 0.'''
        
        MP: The above doesn't make sense for the record case. The trigger can never be in the center of the record in 
         record mode since the record can be way longer than the buffer. Let's just figure out when the trigger is
          at the very beginning of the trace.
        """ 
        #print("Trigger Position Possible Values:{}".format(analogIn.triggerPositionInfo()))
        #print("MP: These trigger position numbers are crazy. There is a bug or the documentation is messed up")
        analogIn.triggerPositionSet(trigger_position)
         
        analogIn.triggerTypeSet(DwfAnalogInTriggerType.Edge)
        analogIn.triggerConditionSet(DwfTriggerSlope.Rise)
       
        analogIn.triggerHysteresisSet(0.10)  # A small amount of hysteresis to make sure we only see rising edges.
        
        # triggering on AnalogOut (basically a TTL) doesn't have a voltage level 
        if analogIn.triggerSourceGet() == DwfTriggerSource.DetectorAnalogIn:
            analogIn.triggerLevelSet(TRIGGER_LEVEL)#[V]
    return channels       

def acquire_one_record_record(analogIn, channels,trigger_flag):
    """"
    This function gathers one record worth of samples ... sometimes in multiple queries to the ADC
    It see's if there are any samples available for collecting and collects them a few at a time
    
    output
       1) record: [nsamples, nchannel] array
       2) time_of_first_sample: trigger time of the record
    """ 
    record=[]

    total_samples_lost = total_samples_corrupted = 0
    number_ADC_queries = 0

    while True:
        number_ADC_queries +=1

        status = analogIn.status(True)
        (
            current_samples_available,
            current_samples_lost,
            current_samples_corrupted,
            ) = analogIn.statusRecord()
        
        total_samples_lost += current_samples_lost
        total_samples_corrupted += current_samples_corrupted

        if current_samples_lost != 0:
            """ 
            Append NaN samples as placeholders for lost samples.
            This follows the Digilent example.
            We haven't verified yet that this is the proper way to handle lost samples.
            """
            lost_samples = np.full((current_samples_lost, len(channels)), np.nan)
            record.append(lost_samples)
        if current_samples_available != 0:
            """"
             Append samples read from both channels.
             Note that we read the samples separately for each channel;
             We then put them into the same 2D array with shape (current_samples_available, len(channels)).
            """
            current_samples = np.vstack(
                [
                    analogIn.statusData(channel_index, current_samples_available)
                    for channel_index in channels
                ]
            ).transpose()
            record.append(current_samples)

        if status == DwfState.Done:
            print("Total # of ADC Queries for this event:{}".format(number_ADC_queries))
            # We received the last of the record samples.
            # Note the time, in seconds, of the first valid sample, and break from the acquisition loop.
                
            #Let's concatenate the record so that is has the correct size
            record = np.concatenate(record)
            
            #This keeps the beginning of the file where the trigger is 
            if len(record) > RECORD_LENGTH_SAMPLES:
                discard_count = len(record) - RECORD_LENGTH_SAMPLES
                # print(
                #     "- NOTE - discarding newest {} of {} samples ({:.1f}%); keeping {} samples.".format(
                #         discard_count,
                #         len(record),
                #         100.0 * discard_count / len(record),
                #         RECORD_LENGTH_SAMPLES,
                #     )
                # )
        
                record = record[:RECORD_LENGTH_SAMPLES]
                
            if trigger_flag:
                time_of_first_sample = analogIn.triggerPositionStatus()
            else:
                time_of_first_sample = 0.0

            break        

    if total_samples_lost != 0:
        print("- WARNING - {} samples were lost! Reduce sample frequency.".format(total_samples_lost))

    if total_samples_corrupted != 0:
        print("- WARNING - {} samples could be corrupted! Reduce sample frequency.".format(total_samples_corrupted))
    return record, time_of_first_sample

def acquire_one_record_single(analogIn, channels,trigger_flag):
    """
    This function is used when analogIn is in the single and single1 mode.
    It   gathers one record worth of samples in one read call to the ADC
    
    Outputs:
      1) record: [nsamples, nchannel] array
      2) time_of_first_sample: trigger time of the record 
    """
    number_ADC_queries = 0

    nSamples_Buffer= analogIn.bufferSizeGet()
    #print("Buffersize: {}".format(nSamples_Buffer))
    
    while True:
        number_ADC_queries +=1

        status = analogIn.status(True)
        #print("status pre done: {}".format(status))
        
        #let's only pull data from the ADC buffer when status= Done
        if status==DwfState.Done:   
            # Note that we read the samples separately for each channel;
            # We then put them into the same 2D array with shape (current_samples_available, 2).
            record = np.vstack(
                [
                    #analogIn.statusData(channel_index, current_samples_available)
                    analogIn.statusData(channel_index,nSamples_Buffer)
                    for channel_index in channels
                ]
            ).transpose()

            #------- We need to concatenate the file to the appropriate size ------
            #This keeps the beginning of the file ... thus the distance from the start of trace to
            #to the trigger will remain fixed. 
            if len(record) > RECORD_LENGTH_SAMPLES:
                discard_count = len(record) - RECORD_LENGTH_SAMPLES
                print(
                    "- NOTE - discarding newest {} of {} samples ({:.1f}%); keeping {} samples.".format(
                        discard_count,
                        len(record),
                        100.0 * discard_count / len(record),
                        RECORD_LENGTH_SAMPLES,
                    )
                )
                record = record[:RECORD_LENGTH_SAMPLES]
            
            # Note the time, in seconds, of the first valid sample, and break from the acquisition loop.    
            if trigger_flag:
                #print("TriggerPositionStatus:{}".format(analogIn.triggerPositionStatus()))
                #print("TriggerPosition:{}".format(analogIn.triggerPositionGet()))
                
                # when trigger Position is zero it means that the trigger is at the center of the trace
                # when trigger position is -Record_Length_Time/2 ... the trigger is at the beginning of the record
                time_of_first_sample= -analogIn.triggerPositionStatus() -Record_Length_Time/2
            else:
                time_of_first_sample = 0.0

            break

    print("Total Number of ADC Queries For This Trace: {}".format(number_ADC_queries))

    return record, time_of_first_sample

def run_acquistion(
    analogIn,
    trigger_flag,
    number_records,
    channels
):
    """"
    Run Acquisition does 2 things:
      1) Records samples/traces
      2) makes and updates plots 
    """
    number_records_taken = 0
    if lgc_lockIn:
        #------- Initialize Lockin --------------------
        time_lockin = np.zeros((number_records,1)) 
        record_lockin = np.zeros((number_records,1))
        time_start_acquisition= time.time()
        
        # Initialize Demodulated vs time plot
        lines_lockin =lines_lockinfft=None
        label_lockin = ['channel 1']
        fig_lockin, ax_lockin = plt.subplots(2, 1, figsize=(5, 8))
        plt.subplots_adjust(hspace=0.5)

    if lgc_demodulate:
        #------- Initialize Demodulated Plots --------------
        # Initialize Demodulated vs time plot
        lines_demod =lines_demodfft=None
        label_demod = ['channel 1']
        fig_demod, ax_demod = plt.subplots(2, 1, figsize=(5, 8))
        plt.subplots_adjust(hspace=0.5)    

    #------- Initialize Record Plots --------------
    # Initialize Raw Voltage vs time plot
    lines_raw=lines_rawfft=None
    label_raw=['channel 1','channel 2','demod line']
    fig_raw, ax_raw = plt.subplots(2, 1, figsize=(5, 8))
    plt.subplots_adjust(hspace=0.5)
    
    while True:
      
        #--------------- 1) Records Traces --------------------------
        print("[{}] Recording {} samples ...".format(number_records_taken, RECORD_LENGTH_SAMPLES))

        """
        analogIn.configure(reconfigure,start)
          reconfigure (bool): 
                  true  = commits all the changes ... this isn't needed since auto-config is default.
                  false = no changes are committed
          start(bool):
                  True = run acquisition
                  False = stop acquistion
        """    
        analogIn.configure(False, True)  # Start acquisition sequence for one record

        #Let's use different single record acquisition functions for the DAQ different modes
        if analogIn.acquisitionModeGet()==DwfAcquisitionMode.Record:
            record,time_of_first_sample = acquire_one_record_record(analogIn, channels,trigger_flag)
        else:
            record,time_of_first_sample = acquire_one_record_single(analogIn, channels,trigger_flag)
        
        # Let's find the time of the samples in the record with respect to first sample
        time_sample = time_of_first_sample + np.arange(len(record)) / ANALOG_IN_SAMPLE_FREQUENCY

        if lgc_demodulate:
            #----------- Demodulate Signal -----------------------------------
            record_demod, record_demod_filt, ref_demod = Demodulate(record, time_sample,ANALOG_OUT_FREQUENCY,PHASE_DEMOD)
            
            #----------- Update Modulated TracePlot ----------
            y_raw=np.concatenate([record,ref_demod[:,np.newaxis]],axis=1)
            title_raw= "AnalogIn acquisition #{}".format(number_records_taken,)
            
            xlim_time=None
            #xlim_time=[0,5/ANALOG_OUT_FREQUENCY]
            lines_raw=plot_time(ax_raw[0],
                lines_raw,        
                y_raw,
                time_sample,
                label_raw,
                title_raw,
                xlim_time     
            )

            CH1=0
            # TODO 10.4: uncomment this line Now, plot the FFT on the second subplot
            xlim_fft=None
            #xlim_fft=[995,1005]
            #xlim_fft=[0,10]
            xlim_fft=[0,ANALOG_OUT_FREQUENCY*5]
            
            lines_rawfft=None #let's have plots from every record on the plot
            lines_rawfft=plot_fft(record[:, CH1], ANALOG_IN_SAMPLE_FREQUENCY, ax_raw[1],lines_rawfft,xlim_fft)
            
            #updated plots
            fig_raw.canvas.draw() 
            fig_raw.canvas.flush_events()
            
            #--- update demodulation plots ----
            title_demod= "Demodulated Trace #{}".format(number_records_taken,)
            xlim_time=None
            
            lines_demod =plot_time(ax_demod[0],
                lines_demod,        
                record_demod_filt[:,np.newaxis],
                time_sample,
                label_demod,
                title_demod,
                xlim_time     
            )
            xlim_fft=None
            lines_demodfft = None  #every record will have its own fft trace
            lines_demodfft =plot_fft(record_demod, ANALOG_IN_SAMPLE_FREQUENCY, ax_demod[1],lines_demodfft,xlim_fft)
            
            #These commands force the figure update
            fig_demod.canvas.draw() 
            fig_demod.canvas.flush_events()
        
        if lgc_lockIn:
            #----------- Lockin Signal -------------------------------
            # Let's find the time of first sample since the start of acquistion 
            time_lockin[number_records_taken]= time.time()-time_start_acquisition
            record_lockin[number_records_taken], ref_demod = LockIn(record, time_sample,ANALOG_OUT_FREQUENCY,PHASE_DEMOD)  

            Lockin_Frequency = (number_records_taken)/(time_lockin[number_records_taken]-time_lockin[0])
            
            #---------- Make/Update Plots -----------------------------
            lgc_diagnostic = False
            if lgc_diagnostic or  (number_records_taken >= number_records-1):
                print("Estimated Lock-In Frequency:{} Hz vs Trace Frequency:{} Hz".format(Lockin_Frequency, 1/Record_Length_Time))

                #---- update lock-in plots ----
                title_lockin= "Lock-in Trace after {} records".format(number_records_taken,)
                
                xlim_lockin=None
                lines_lockin = plot_time(ax_lockin[0],
                    lines_lockin,        
                    record_lockin[:number_records_taken+1,:],
                    time_lockin[:number_records_taken+1,:],
                    label_lockin,
                    title_lockin,
                    xlim_lockin     
                )
                
                #print("time_lockin:".format(time_lockin[number_records_taken,0]))
                #print("record_lockin:".format(record_lockin[number_records_taken,0]))

                if number_records_taken>10:
                    # let's make an FFT of lockin- data
                    # Let's assume that the lockin data points are roughly evenly spaced ... 
                    # this is true on my machine. If you don't make this assumption, calculating
                    # the fourier transform is way, more complex. 
                    
                    #xlim_fft=None
                    xlim_fft=[0.25,2]
                    lines_lockinfft=plot_fft(record_lockin[:number_records_taken+1,0], Lockin_Frequency, ax_lockin[1],lines_lockinfft,xlim_fft)
                
                fig_lockin.canvas.draw() 
                fig_lockin.canvas.flush_events() 

                #----------- Update Single Trace Plots ------------------------------
                y_raw=np.concatenate([record,ref_demod[:,np.newaxis]],axis=1)
                title_raw= "AnalogIn acquisition #{}".format(number_records_taken,)

                #xlim_time=None
                xlim_time=[0,5/ANALOG_OUT_FREQUENCY]
                lines_raw=plot_time(ax_raw[0],
                    lines_raw,        
                    y_raw,
                    time_sample,
                    label_raw,
                    title_raw,
                    xlim_time     
                )

                CH1=0
                # TODO 10.4: uncomment this line Now, plot the FFT on the second subplot
                xlim_fft=None
                #xlim_fft=[995,1005]
                #xlim_fft=[0,10]
                xlim_fft=[0,ANALOG_OUT_FREQUENCY*5]
                
                lines_rawfft=plot_fft(record[:, CH1], ANALOG_IN_SAMPLE_FREQUENCY, ax_raw[1],lines_rawfft,xlim_fft)
                
                #updated plots
                fig_raw.canvas.draw() 
                fig_raw.canvas.flush_events()

        if lgc_test: #just the Modulated trace plot
            #----------- Update Modulated TracePlot ----------
            title_raw= "AnalogIn acquisition #{}".format(number_records_taken,)
            
            xlim_time=None
            #xlim_time=[0,5/ANALOG_OUT_FREQUENCY]
            lines_raw=plot_time(ax_raw[0],
                lines_raw,        
                record,
                time_sample,
                label_raw,
                title_raw,
                xlim_time     
            )

            #///////////// Plot FFT ///////////////
            #Student Version commented out
            # TODO 10.4: everything between the "///" lines for the fft plot
            CH1_FFT=0
            xlim_fft=None
            #xlim_fft=[995,1005]
            #xlim_fft=[0,10]
            #xlim_fft=[0,ANALOG_OUT_FREQUENCY*5]
            lines_rawfft=None #let's have plots from every record on the plot

            lines_rawfft=plot_fft(record[:, CH1_FFT], ANALOG_IN_SAMPLE_FREQUENCY, ax_raw[1],lines_rawfft,xlim_fft)
            #////////////////////////////////////

            #matplotlib commands to update plots 
            fig_raw.canvas.draw() 
            fig_raw.canvas.flush_events()


        #increment acquisition number
        number_records_taken += 1  # Increment acquisition number.
        
        if (len(plt.get_fignums()) == 0) or (number_records_taken >= number_records):
            break

def LockIn(record, time_sample, ANALOG_OUT_FREQUENCY, PHASE_DEMOD) :
    """
    # This function:
    # 1) Calculates the demodulation phase angle from the excitation ... if not given. 
    # 2) makes the demodulation wave -> the phase needs to be checked in the real time plots
    # 3) demodulates the signal
    # 4) calculates the DC lock-in value
    """

    phi= ANALOG_OUT_FREQUENCY*(2*np.pi)*time_sample

    #------ 1) Calculate the demodulation phase angle
    # Let's calculate the demodulation phase angle for the reference signal 
    # if the trigger was behaving at is should ... we shouldn't need to do this
    #we're basically just taking the FFT at the excitation frequency
    if PHASE_DEMOD is None:
        PHASE_DEMOD = np.angle(np.sum(record[:,1]*(np.cos(phi)-1j*np.sin(phi)))) #[radians]
        print("Estimated Demodulation angle: {} deg".format(PHASE_DEMOD*180/np.pi))
    


    #------ 2) Make the Demodulation Wave ------
    ref_demod= np.cos(phi+PHASE_DEMOD)
    
    #//////////////////////////////////////////////////////
    #------ 3) Demodulate the Signal -----------
    #TODO: 10.7.c make the demodulated signal (just like you did in 10.6) 
    #Initial Student Version
    CHsignal = 0
    record_demod = record[:,CHsignal]

    #Solution here
    
    #------ 4) Calculate the DC lock-in --------
    #TODO: 10.7.c: make the average
    #Initial Student Version
    record_DC_Lockin = 0
    
    #Solution here
    #////////////////////////////////////////////////////////

    return record_DC_Lockin,ref_demod

          
def Demodulate(record, time_sample, ANALOG_OUT_FREQUENCY,PHASE_DEMOD):
    """
    This function:
    1) Calculates the demodulation phase angle from the excitation (if necessary)
    2) makes the demodulation wave -> the phase needs to be checked in the real time plots
    3) demodulates the signal
    4) Lowpass filter to remove 2*nu_carrier components
    """

    # Let's calculate the demodulation phase angle for the reference signal -> we shouldn't need to do this
    phi= ANALOG_OUT_FREQUENCY*(2*np.pi)*time_sample

    #----- 1) Calculate Demodulation Phase ------- 
    if PHASE_DEMOD is None:
        #we're basically just taking the FFT at the excitation frequency
        PHASE_DEMOD = np.angle(np.sum(record[:,1]*(np.cos(phi)-1j*np.sin(phi)))) #[radians]
        print("Estimated Demodulation angle: {} deg".format(PHASE_DEMOD*180/np.pi))

    #----- 2) MAKE DEMODULATION FUNCTION ------
    #Let's create demodulation function
    ref_demod= np.cos(phi+PHASE_DEMOD)
    
    #///////////////////////////////////////////////////////////////
    #----- 3) DEMODULATE SIGNAL ---------------
    # TODO: 10.6f write the code to demodulate record
    # Student initally sees
    CH_demod=0 
    record_demod = record[:,CH_demod]

    #Solution ?
    #///////////////////////////////////////////////////////////////

    #----- 4) Low Pass Filter -----------------
    #Let's put a simple 1pole low pass filter on it. 
    #You can also use the scipy python package to get more advanced filters
    
    #let's build a simple filter that is at 20 Hz
    nu_3dbfilter= 20 #[Hz] 
    #how many samples need to be average to give a  20Hz rolloff?
    n_filt = round(nu_3dbfilter*Record_Length_Time)
    if n_filt>1:
        record_demod_filt= np.convolve(record_demod, np.ones(n_filt)/n_filt, mode='same')
    else:
        record_demod_filt=record_demod #no filter applied

    return record_demod,record_demod_filt,ref_demod


def plot_time(
    ax0,    
    lines,
    y,  #np.array(number of x points, number of plots)
    x,  #np.array
    label_raw,
    title_raw,
    xlim_time
):
    
    color_str=["b","g","r","y","m"]

    n_lines= np.shape(y)[1]

    # Subplot 1: Voltage vs time
    if lines is None:
        # This is the first time we plot acquisition data.
        ax0.grid()
        ax0.set_xlabel("time relative to trigger [s]")
        ax0.set_ylabel("signal [V]")

        # 9.1.3: uncomment the line below and comment out the line above
        # plt.ylim(-1.1 * signal_amplitude + 1.9, +1.1 * signal_amplitude + 1.9)

        lines = ax0.plot(x, y[:, 0], color=color_str[0], label=label_raw[0])
        for j_line in range(1,n_lines):
            lines.extend(
                    ax0.plot(x, y[:, j_line], color=color_str[j_line], label=label_raw[j_line])
            )
        ax0.legend(loc="upper right")
    else:
        # The plot is already available. Just update the acquisition data.
        for j_line in range(n_lines):
            lines[j_line].set_xdata(x)
            lines[j_line].set_ydata(y[:,j_line])

    if xlim_time is None:
        xmin=x.min()
        xmax=x.max()
        ax0.set_xlim((xmin-0.1*np.abs(xmin), xmax+0.1*np.abs(xmax)))
    else:
        ax0.set_xlim(xlim_time)

    #ax0.set_ylim(-7, 7)
    ymax=y.max()
    ymin=y.min()
    
    ax0.set_title(title_raw)

    return lines


def plot_fft(trace, sample_frequency, ax_fft,lines, xlim_fft):
    #/////////////////////////////////////////////////////////////////////////
    # 10.4b TODO: get the FFT Frquency (fft_freq) and magnitude (fft_magnitude). 
    #--- Initial Student Version place holder ---
    fft_freq = fft_magnitude = np.zeros(len(trace))
    #----- Solution-----
    
    # END TODO //////////////////////////////////////////////////////////

    #select those frequencies that are positive for plotting
    lgc_positive = fft_freq >= 0

    #----------- Plot the magnitude of the FFT vs frequency -------------
    # Plotting the FFT on the provided axis
    # By convention only plot the positive frequencies
    if lines is None:
        ax_fft.grid()
        ax_fft.plot(fft_freq[lgc_positive], fft_magnitude[lgc_positive])
        #ax_fft.set_xscale("log")
        #ax_fft.set_yscale("log")
        ax_fft.set_title("FFT of the signal")
        ax_fft.set_xlabel("Frequency [Hz]")
        ax_fft.set_ylabel("Magnitude")
    else:
        lines.set_xdata= fft_freq[lgc_positive]
        lines.set_ydata= fft_magnitude[lgc_positive]       
   
    if xlim_fft is None:
        xlim_fft = [0, sample_frequency / 2]
        lgc_range= np.full((len(trace)),True)
    else:
        lgc_range= np.logical_and(fft_freq<xlim_fft[1],fft_freq>xlim_fft[0])
    
    ax_fft.set_xlim(xlim_fft)    
    
    #lgc_range= lgc_positive
    lgc_range= np.logical_and(lgc_range,lgc_positive)
    ymax=2.0*fft_magnitude[lgc_range].max()
    #ax_fft.set_ylim(None,ymax)  # Adjust if needed
    #ax_fft.set_ylim((0,ymax))  # Adjust if needed
    ax_fft.set_ylim(0,ymax)

    return lines    

    



def main():
    dwf = DwfLibrary()

    def maximize_analog_in_buffer_size(configuration_parameters):
        """ In waveforms there are the Device Manager Settings. This function looks at
        the configuration parameter and outputs the maximum AnalogIn (Oscilloscope) buffersize of each configuration option
        
        when sent to OpenDwfDevice as the score_func input, OpenDwfDevice will loop through all
        configurations and  find the one that maximizes this value
        
        See this link for more info on how openDwfDevice works: https://pydwf.readthedocs.io/en/latest/pydwf_api/pydwf_utilities.html"""
        return configuration_parameters[DwfEnumConfigInfo.AnalogInBufferSize]

    try:
        # See this link for more info on how openDwfDevice works: https://pydwf.readthedocs.io/en/latest/pydwf_api/pydwf_utilities.html
        with openDwfDevice(
            dwf,
            score_func=maximize_analog_in_buffer_size, #see comment above to see how score func works
        ) as device:
            #This is the object to control wavegen
            analogOut = device.analogOut
            
            #This is the object to control the Oscilloscope/ADC
            analogIn = device.analogIn

            #---- WaveGen Setup ----
            print(
                "Configuring analog output signals ({} Hz) ...".format(ANALOG_OUT_FREQUENCY)
            )

            configure_analog_out(analogOut, ANALOG_OUT_FREQUENCY)
            print("starting analog output:")
            
            time.sleep(2.0) # Wait for a bit to ensure the stability of the analog output signals.

            # 10.3.a: TODO: comment out this while loop which pauses the loop before getting to the oscilloscope.
            while True:
                time.sleep(2.0)

            #----- Configure analogIn (Oscilloscope) ------
            USE_TRIGGER= True
            channels=configure_analog_in(analogIn, USE_TRIGGER)

            #----- Acquire data and plot individual traces ----
            #turn on interactive mode
            plt.ion()

            run_acquistion(
                analogIn,
                USE_TRIGGER,
                NUMBER_RECORDS,
                channels
            )
            
            #Let's pause so that we can interact with the plots!
            print("Close plts to finish program")
            while (len(plt.get_fignums()) != 0):
                plt.pause(1)
            

    except PyDwfError as exception:
        print("PyDwfError:", exception)


if __name__ == "__main__":
    main()
