#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 09:53:51 2021

@author: antonfredriksson
"""

from mpi4py import MPI
""" Get a communicator :
The most common communicator is the
one that connects all available processes
which is called COMM_WORLD
"""
# acts like COMM_WORLD but is a separate instance
comm = MPI . Comm . Clone ( MPI . COMM_WORLD )
# print rank ( the process number ) and overall number of processes
print ("Hello World : process", comm.Get_rank() , "out of", comm.Get_size() , "is reporting for duty !")