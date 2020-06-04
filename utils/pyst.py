#!/usr/bin/env python
#
# file: $(NEDC_NFC)/util/python/nedc_pystream/nedc_pystream.py
#
# revision history:
#  20190715 (JP): rewrote the utility and cleaned up the code
#  20190712 (MM): fixed formatting errors and load_edf
#  20190705 (JP): code review
#  20190703 (NS): code review
#  20190616 (SJP & MM): initial version
# 
# usage:
#   nedc_pystream -parameters parameters foo.edf
#
# options:
#  -parameters: a parameter file 
#  -help: display this help message
#
# arguments: takes a single EDF file
#
# This script takes in an EDF file and outputs the samples of all channels
# depending on the specified parameters.
#
#------------------------------------------------------------------------------

# import system modules:
#  Most of the modules listed below are found in standard Python distributions 
#  such as Anaconda (which we use). However, one external library you will 
#  need is pyedflib. This library can be downloaded from GitHub:
#
#   https://github.com/holgern/pyedflib
#
import os
import re
import sys
import time
import argparse
import pyedflib
from collections import OrderedDict
import numpy as np
 
# parameter file constants:
# these are reserved symbols used to parse paramter files
#
DELIM_BCLOSE = '}'
DELIM_BOPEN = '{'
DELIM_CLOSE = ']'
DELIM_COLON = ':'
DELIM_COMMA = ','
DELIM_COMMENT = '#'
DELIM_EQUAL = '='
DELIM_NEWLINE = '\n'
DELIM_NULL = ''
DELIM_QUOTE = '"'
DELIM_SPACE = ' '
DELIM_SUB = "--"

# parameter file constants:
# these are reserved words used to parse paramter files
#
KEYWORD_VERSION = "version"
KEYWORD_VERSION_NUMBER = "1.0"
KEYWORD_HELP = "-help"

KEYWORD_CSEL = "channel_selection"
KEYWORD_MONTAGE = "montage"
KEYWORD_MMODE = "match_mode"

KEYWORD_NULL = "(null)"
KEYWORD_EXACT = "exact"
KEYWORD_PARTIAL = "partial"

#******************************************************************************
#
# functions start here
#
#******************************************************************************

#------------------------------------------------------------------------------
# function: nedc_print_usage
#
# arguments: none
#
# return: none
#
# This function prints a usage message to stdout.
#
def nedc_print_usage():
     print ("Usage: nedc_pystream [-help] -p pfile.txt file1.edf")
     exit(-1)
#
# end of function

#------------------------------------------------------------------------------
# function: nedc_print_help
#
# arguments: none
#
# return: none
#
# this function prints a help message to stdout.
#
def nedc_print_help():
    print ("name: nedc_pystream")
    print ("synopsis: nedc_pystream [options] file")
    print ("descr: demonstrates how to read an Edf file")
    print ("")
    print ("options:")
    print (" -parameters: a feature extraction parameter file")
    print (" -help: display this help message")
    print ("")
    print ("arguments:")
    print (" file: a single EDF file")
    print ("")
    print ("examples:")
    print ("")
    print (" nedc_pystream -p params.txt file1.edf")
    print ("")
    print ("  prints the signal data to stdout as floating point numbers")
    print ("")
    print (" nedc_pystream -help -param params.txt file1.edf")
    print ("")
    print ("  displays this help message")
    print ("")
    print ("notes:")
    print ("")
    print (" the montage specified in the parameter file controls the order")
    print (" and type of data printed")
    exit(-1)
#
# end of function

#------------------------------------------------------------------------------
# function: nedc_print_vals
#
# arguments: 
#   fsamp: the sample frequencies for each channel
#   sig: the signal data
#   labels: the channel labels
#
# return: 
#   status: the status of the function
#
# this method prints the values of the signals to stdout
#
def nedc_print_vals(fsamp_a, sig_a, labels_a):

     # check for no data
     #
     if len(sig_a) == int(0):
          print ("%s (%s: %s) empty signal" \
            % (sys.argv[0], __name__, "nedc_print_vals"))
          return True

     # loop over all time and then by channel - use the minimum length channel
     #
     nchans = len(sig_a)
     nsamples = min(map(len, sig_a))
     print ("no. output channels = %3d" % (nchans))
     for i in range(nsamples):
          print ("sample no. %4d: " % (i))
          for j in range(len(sig_a)):
               print (" channel %3d (%15.15s, %12.4f): %12.4f" % \
                    (j, labels_a[j], fsamp_a[j], sig_a[j][i]))

     # exit gracefully
     #
     return True
#
# end of function

#------------------------------------------------------------------------------
# function: nedc_load_parameters
#
# arguments: 
#   pfile: parameter file
#
# return:
#   values: an ordered dictionary that contains the name/value pairs in
#           the parameter file
#
# This function loads a parameter file and returns the associated name/value
# pairs in a dictionary data structure. Note that the montage specification
# is stored as one entry (a list) in the dictionary.
#
def nedc_load_parameters(pfile_a):

     # declare local variables
     #
     values = OrderedDict()
     keyword_upcase = KEYWORD_MONTAGE.upper()

     # open the file
     #
     try:
          fp = open(pfile_a, "r")
     except:
          print ("%s (%s: %s): file not found (%s)" \
               % (sys.argv[0], __name__, "nedc_load_parameters", pfile_a))
          return None

     # loop over all lines in the file
     #
     flag_pblock = False
     flag_montage = False
     for line in fp:
          
          # clean up the line
          #
          str = line.replace(DELIM_SPACE, DELIM_NULL)
          str = str.replace(DELIM_NEWLINE, DELIM_NULL)

          # throw away commented and blank lines
          #
          if (str.startswith(DELIM_COMMENT) == True) or (len(str) == 0):
               pass

          # check for the version
          #
          elif str.startswith(KEYWORD_VERSION) == True:
               parts = str.split(DELIM_EQUAL)
               if parts[1] != KEYWORD_VERSION_NUMBER:
                    print ("%s (%s: %s): incorrect version number (%s)" \
               % (sys.argv[0], __name__, "nedc_load_parameters", parts[1]))
                    return None

          # check for the beginning of a parameter block
          #
          elif (str.startswith(keyword_upcase) == True) and \
               (DELIM_BOPEN in str):
               flag_pblock = True

          # check for the end of a parameter block:
          # note that we exit if we hit the end of the parameter block
          #
          elif (flag_pblock == True) and (DELIM_BCLOSE in str):
               fp.close();
               break;

          # otherwise, if the parameter block has started, decode a parameter
          # by splitting and assigning to a dictionary
          #
          elif (flag_pblock == True):
               parts = str.split(DELIM_EQUAL)

               # check for the first occurrence of a montage entry and
               # initialize a list
               #
               if (parts[0] == KEYWORD_MONTAGE) and (flag_montage == False):
                    values[parts[0]] = []
                    flag_montage = True
               
               # if it is a montage keyword: append the montage list
               #
               if (parts[0] == KEYWORD_MONTAGE):
                    values[parts[0]].append(parts[1].replace(
                         DELIM_QUOTE, DELIM_NULL))

               # else: treat it as a normal name/value pair
               #
               else:
                    values[parts[0]] = parts[1].replace(
                         DELIM_QUOTE, DELIM_NULL)

     # close the file pointer
     #
     fp.close()

     # make sure we found a block
     #
     if flag_pblock == False:
          print ("%s (%s: %s): invalid parameter file (%s)" \
               % (sys.argv[0], __name__, "nedc_load_parameters", pfile_a))
          return None

     # exit gracefully
     #
     return values
#
# end of function

#------------------------------------------------------------------------------
# function: nedc_load_edf
#
# arguments: 
#   fname: filename (input)
#
# return: 
#   labels: store the EDF signal labels
#   fsamp: store the EDF signal sample frequency
#   sig: signals in the EDF file
#
# this function loads the EDF and return the signals
#
def nedc_load_edf(fname_a):

    # open an EDF file 
    #
    try:
        fp = pyedflib.EdfReader(fname_a)
    except IOError:
        print ("%s (%s: %s): failed to open %s" % \
            (sys.argv[0], __name__, "nedc_load_edf", fname_a))
        exit(-1)

    # get the metadata that we need:
    #  convert the labels to ascii and remove whitespace 
    #  to make matching easier
    #
    num_chans = fp.signals_in_file
    labels_tmp = fp.getSignalLabels()
    labels = [str(lbl.replace(' ', '')) for lbl in labels_tmp]

    # load each channel
    #
    sig = []
    fsamp = []
    for i in range(num_chans):
        sig.append(fp.readSignal(i))
        fsamp.append(fp.getSampleFrequency(i))

    # exit gracefully
    #
    return (fsamp, sig, labels)
#
# end of function 

#------------------------------------------------------------------------------
# function: nedc_get_pos
#
# arguments:
#   lbl: label to be located
#   labels: list of labels
#   mmode: match mode
#
# return:
#   pos: the position in the list
#
# This function locates a label on the list and returns the position.
#
def nedc_get_pos(lbl_a, labels_a, mmode_a):

    # declare local variables
    #
    indices = []

    # mode: exact
    #  note that we return the first match
    #
    if mmode_a == KEYWORD_EXACT:
        pos = labels_a.index(lbl_a)
        if pos >= int(0):
            indices.append(pos)
        else:
            indices.append(int(-1))

    # mode: partial
    #
    else:
        indices = [i for i, elem in enumerate(labels_a) if lbl_a in elem]

    # exit gracefully
    #
    if len(indices) == 0:
        return int(-1)
    else:
        return indices[0]
    
#------------------------------------------------------------------------------
# function: nedc_select_channels
#
# arguments:
#   params: parameter block dictionary
#   fsamp: the sample frequencies for each channel
#   sig: the signal data
#   labels: the channel labels
#
# return:
#   fsamp_sel: output sample frequency list
#   sig_sel: output signal data
#   labels_sel: output channel labels
#
# This function returns selects channels from a signal and returns a
# subset of the channels.
#
def nedc_select_channels(params_a, fsamp_a, sig_a, labels_a):
    
    # declare local variables
    #
    fsamp_sel = []
    sig_sel = []
    labels_sel = []

    # extract the list of channels from the parameter block
    #
    chan_list = params_a.get(KEYWORD_CSEL).split(DELIM_COMMA)
    print ('nedc_select_channels', chan_list)
    
    # if the channel list contains null, simply copy the input to the output
    #
    if KEYWORD_NULL in chan_list:
        return (fsamp_a, sig_a, labels_a)

    # else: copy selected channels
    #
    for lbl in chan_list:

        # look up the label in the original signal
        #
        pos = nedc_get_pos(lbl, labels_a, params_a[KEYWORD_MMODE])

        # append the corresponding signal
        #
        if pos >= int(0):
            fsamp_sel.append(fsamp_a[pos])
            sig_sel.append(sig_a[pos])
            labels_sel.append(labels_a[pos])
        else:
            print ("%s (%s: %s): failed to find label %s" % \
                (sys.argv[0], __name__, "nedc_select_channels", lbl))
            exit(-1)

    # exit gracefully
    #
    return fsamp_sel, sig_sel, labels_sel
#
# end of function 

#------------------------------------------------------------------------------
# function: nedc_parse_montage
#
# arguments:
#   params: parameter block dictionary
#
# return:
#   montage: a list of n-tuples containing a montage specification
#
# This function converts the montage in the parameter block to a more
# user-friendly data structure.
#
def nedc_parse_montage(params_a):

     # loop over all montage entries:
     #  build a list that contains the channel index, the output label,
     #  the first input channel, and the second channel if a difference
     #  is specified.
     #
     montage = []

     for str in params_a[KEYWORD_MONTAGE]:

          # split the line into two pieces: the channel index
          # and the labels
          #
          parts = str.split(DELIM_COMMA)          

          # split the right-hand side into output label and input label
          #
          subparts = parts[1].split(DELIM_COLON)

          # split the input label into two terms if a difference is specified
          #
          expparts = subparts[1].split(DELIM_SUB)

          # assemble it into a full list
          #
          parts[1] = subparts[0]
          parts.append(expparts[0])
          if len(expparts) > 1:
               parts.append(expparts[1])
          else:
               parts.append(KEYWORD_NULL)
          montage.append(parts)

     # exit gracefully
     #
     return montage
#
# end of function

#------------------------------------------------------------------------------
# function: nedc_apply_montage
#
# arguments:
#   params: parameter block dictionary
#   fsamp: the sample frequencies for each channel
#   sig: the signal data
#   labels: the channel labels
#
# return:
#   fsamp_mont: output sample frequency list
#   sig_mont: output signal data
#   labels_mont: output channel labels
#
# This function applys a montage to a signal.
#
def nedc_apply_montage(params_a, fsamp_a, sig_a, labels_a):

     # initialize the output variables
     #
     fsamp_mont = []
     sig_mont = []
     labels_mont = []

     # if the montage specification contains null, simply copy the 
     # input to the output
     #
     if KEYWORD_NULL in params_a[KEYWORD_MONTAGE.lower()]:
          return (fsamp_a, sig_a, labels_a)

     # convert the raw format of the montage into something
     # that is easier to process
     #
     montage = nedc_parse_montage(params_a)

     # loop over the output montage
     #
     for i in range(len(montage)):

          # get the position of the first operand
          #
          pos1 = nedc_get_pos(montage[i][2], labels_a,
                              params_a[KEYWORD_MMODE])
          if montage[i][3] != KEYWORD_NULL:
               pos2 = nedc_get_pos(montage[i][3], labels_a,
                                   params_a[KEYWORD_MMODE])
          else:
               pos2 = int(-1)

          # compute the new length as the shorter of the two
          #
          min_len = len(sig_a[pos1])
          if (pos2 >= int(0)):
               if len(sig_a[pos2]) < min_len:
                    min_len = len(sig_a[pos2])

          # copy the first signal
          #
          sig_mont.append(sig_a[pos1])
          sig_mont[i] = sig_mont[i][:min_len]

          # difference the two signals if necessary
          #
          if pos2 >= int(0):
               for j in range(min_len):
                    sig_mont[i][j] -= sig_a[pos2][j]

          # append the metadata
          #
          fsamp_mont.append(fsamp_a[pos1])
          labels_mont.append(montage[i][1])
    
     # exit gracefully
     #
     return (fsamp_mont, sig_mont, labels_mont)
#
# end of function

#******************************************************************************
#
# the main program starts here
#
#******************************************************************************

#------------------------------------------------------------------------------
# function: main
#
# arguments: none
#
# return: none
#
# this function is the main program.
#
def read_edf(edf_fn, parameters = "params_04_19.txt"):   

     # loads the parameter file
     #
     params = nedc_load_parameters(parameters)
     if params == None:        
          exit(-1)

     # loads the Edf into memory
     #
     fsamp, sig, labels = nedc_load_edf(edf_fn)

     # select channels from parameter file
     #
     fsamp_sel, sig_sel, labels_sel = nedc_select_channels(params,
                                                            fsamp, sig, labels)

     # apply a montage
     #
     fsamp_mont, sig_mont, labels_mont = nedc_apply_montage(
          params, fsamp_sel, sig_sel, labels_sel)

     # print the values to stdout
     #
     #     nedc_print_vals(fsamp_mont, sig_mont, labels_mont)
     assert fsamp_mont[0] == np.mean(fsamp_mont)
     # print (fsamp_mont[0])
     # print (len(sig_mont), sig_mont[0].shape)
     # print (labels_mont)
     
     return fsamp_mont[0], np.array(sig_mont)

def read_edf_elec(edf_fn, parameters = "params_common_electrodes.txt"):

    # loads the parameter file
    #
    params = nedc_load_parameters(parameters)
    if params == None:        
        exit(-1)

    # loads the Edf into memory
    #
    fsamp, sig, labels = nedc_load_edf(edf_fn)

    # select channels from parameter file
    #
    fsamp_sel, sig_sel, labels_sel = nedc_select_channels(params,
                                                        fsamp, sig, labels)

  
    # print (len(labels))
    # print (labels)
    # print (len(labels_sel))
    # print (labels_sel)
     
    return fsamp_sel[0], np.array(sig_sel)


def read_edf_ekg(edf_fn, parameters="params_ekg.txt"):
    # loads the parameter file
    #
    params = nedc_load_parameters(parameters)
    if params == None:
        exit(-1)

    # loads the Edf into memory
    #
    fsamp, sig, labels = nedc_load_edf(edf_fn)
    print(labels)
    # select channels from parameter file
    #
    fsamp_sel, sig_sel, labels_sel = nedc_select_channels(params,
                                                          fsamp, sig, labels)

    # print (len(labels))
    # print (labels)
    # print (len(labels_sel))
    # print (labels_sel)

    return fsamp_sel[0], np.array(sig_sel)


# begin gracefully
#
if __name__ == "__main__":
    parameters = "params_04_19.txt"
    edf_fn = '/mnt/data4/datasets/seizure/TUH_EEG_Seizure/v1.5.1.mini/edf/train/01_tcp_ar/000/00000077/s003_2010_01_21/00000077_s003_t000.edf'
    # fsamp, data = read_edf(edf_fn, parameters)
    fsamp, data = read_edf_elec(edf_fn)
    print (data.shape, fsamp)
#
# end of file 
