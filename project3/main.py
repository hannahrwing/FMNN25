#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 10:36:08 2021

@author: antonfredriksson
"""
from mpi4py import MPI
from project3 import get_matrix_domain_1
 

#####     initial case   #####

#processor 1
# calculate u0 in room 2 with default gamma 1 and gamma 2
# send neumann to processor 0 and 2

#processor 0 and 2
# recieve neumann from processor 1
# calculate u0 in room 1 and room 3 with neumann from processor 1
# send temperatures at gamma 1 and gamma 2 to processor 1

#####    loop    ####

#processor 1
# extract temp. at gamma 1 and gamma 2 from previous u
# calculate u(k+1) in room 2
# send neumann to processor 0 and 2

#processor 0 and 2
# recieve neumann from 0 and 2
# calculate u(k+1) in room 1 and 3 with neumann
# send all temperatures to processor 1

#processor 1
# recieve all temperatures from processors 0 and 2
# do relaxation


if __name__ == "__main__":
    
    comm = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    
    print(rank)
