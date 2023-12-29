# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 00:05:22 2023

@author: James Yan

This is the main script. You can specify the date you'd like to run
"""

import json
from SIMMCalculator import SIMMCalculator
import os
import time

# specify the date you want to summarize SIMM margin
date = '2022-02-07'

start = time.time()
dir_ = os.getcwd()
with open(os.path.join(dir_,'config.json'),'r') as config_file:
    config = json.load(config_file)

# Initialize calculator
prod_list = []
for prod in ['Equity','Credit','Commodity','RatesFX']:
    if config['prod_'+prod]:
        prod_list.append(prod)
rt_list = []
for rt in ['IR','Credit Qualifying','Credit Non-Qualifying','Equity','FX','Commodity']:
    if config['risktype_'+rt]:
        rt_list.append(rt)
simm = SIMMCalculator(date,dir_,prod_list,rt_list,config['sec_file_name'],config['input_file_name'],config['SIMM_version'])
simm.initialize()
# clean data
simm.data_cleaning()
# calculate and aggregate
simm.run_simm()

finish = time.time()
print((finish-start)/60)
