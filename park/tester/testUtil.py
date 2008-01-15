#!/usr/bin/env python

"""
Set the environments for theory.pdf.
"""
#################################################################
import os, sys
#################################################################

#################################################################
## environment name for home directory of PARK
PARK_PATH = 'parkPath'

## the subdirectories that depends on
DEP_DIRS = ('xmlUtil', 'fit', 'optimizer', 'theory/pdf', )

## directory for testing output
EX_BASE_DIR = '../../examples/pdf/'

## constant to control the choice for testing
CHOICE = 1

## constant to control output verbisity for testing
VERBISITY = 2
##################################################################
#################################################################
def SetEnviron():
    """ Set the environment."""
    if PARK_PATH not in os.environ:
        try:
            pdir =  os.path.dirname(os.path.dirname(
                 os.path.dirname(os.path.abspath(__file__))))
        except:
            # pdir = os.getcwd()  # This may not work, so raise the exception.
            raise EnvironmentError, \
                 'environment "park" is not found or set'
    else:
        pdir = os.environ[PARK_PATH]    
    
    if pdir not in sys.path:
        sys.path.append(pdir)
        
    for d0 in DEP_DIRS:
        fullDir = os.path.join(pdir, d0)
        if os.path.isdir(fullDir) and fullDir not in sys.path:
            sys.path.append(fullDir) 
            
    if not os.path.exists(EX_BASE_DIR):
        os.makedirs(EX_BASE_DIR)               
#################################################################

###############     EOF   #######################################         
        
