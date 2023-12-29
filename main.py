# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 20:06:32 2023

@author: yanh
"""

import argparse
import json
from SIMMCalculator import SIMMCalculator
import os

def parse_arguments():
    # allow user to specify the date
    parser = argparse.ArgumentParser(description='Run SIMM Calculator')
    parser.add_argument('--date', help='Specify the date in format YYYY-MM-DD')
    return parser.parse_args()

def main():
    args = parse_arguments()
    dir_ = os.getcwd()
    with open(os.path.join(dir_,'config.json'),'r') as config_file:
        config = json.load(config_file)
    # override date using command line 
    if args.date:
        config['date'] = args.date
    
    # Initialize calculator
    prod_list = []
    for prod in ['Equity','Credit','Commodity','RatesFX']:
        if config['prod_'+prod]:
            prod_list.append(prod)
    rt_list = []
    for rt in ['IR','Credit Qualifying','Credit Non-Qualifying','Equity','FX','Commodity']:
        if config['risktype_'+rt]:
            rt_list.append(rt)
    simm = SIMMCalculator(config['date'],dir_,prod_list,rt_list,config['sec_file_name'],config['input_file_name'],config['SIMM_version'])
    simm.initialize()
    # clean data
    simm.data_cleaning()
    # calculate and aggregate
    simm.run_simm()
    
    
if __name__ == "__main__":
    main()