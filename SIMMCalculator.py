# SIMM calculator

import pandas as pd
import os
import csv
import collections
from collections import defaultdict
import numpy as np
import math
from RiskTypes import IR, Equity, CreditQ, FX, Commodity

class SIMMCalculator:
    def __init__(self,date,file_path,prod_list,risktype_list,sec_file_name,input_file_name,SIMM_version):
        self.date = date
        self.file_path = file_path
        self.prod_list = prod_list
        self.risktype_list = risktype_list
        self.SIMM_version= str(SIMM_version)
        self.version_parameter_path = os.path.join(file_path,"SIMM Data","SIMM Data_"+self.SIMM_version)
        self.input_sec_list_file = os.path.join(file_path,"Input",f"{sec_file_name}_{date}.csv")
        self.input = pd.read_csv(os.path.join(file_path,"Input",f"{input_file_name}_{date}.csv"), keep_default_na=False)
        # the correlation matrix is in the order of ['IR','Credit Qualifying','Credit Non-Qualifying','Equity','Commodity','FX']
        self.risktype_ind = {'IR':0,'Credit Qualifying':1,'Credit Non-Qualifying':2,'Equity':3,'Commodity':4,'FX':5}
        self.CORR_r = np.loadtxt(open(os.path.join(self.version_parameter_path,"corr across risktypes.csv"),"rb"),delimiter=",")
        self.output_path = os.path.join(file_path,"Output")
        self.df_output = pd.DataFrame(columns=(['CP','Prod','Risk Type','DeltaMargin','VegaMargin','CurvatureMargin','BaseCorrMargin','Margin_Prod']))
        self.risktypes_instance = {'Equity':Equity(),'Credit Qualifying':CreditQ(),'IR':IR(),'FX':FX(), 'Commodity':Commodity()}

    def initialize(self):
        for risktype in self.risktype_list:
            self.risktypes_instance[risktype].initialize(self.version_parameter_path)

    def data_cleaning(self):
        sec_id = []
        with open(self.input_sec_list_file, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sec_id.append(row['trade_id'])
        self.input.fillna(0)
        self.input = self.input[self.input['counterparty_id'] != '']
        self.input = self.input[self.input['trade_id'].isin(sec_id)]
        self.input['amount'] = self.input['amount'].astype(float)
        self.input['bucket'] = self.input['bucket'].apply(lambda x:str(x) if x == 'Residual' or x == '' else str(int(x)))

    def run_simm(self):
        cp_list = list(self.input['counterparty_id'].unique())
        ## SIMM calculation engine
        Margin_cp = defaultdict(float)
        Margin_prod = defaultdict(lambda: defaultdict(float))
        print(f'SIMM {self.SIMM_version} Engine Started...')
        ## begin iterate through each cp, and then prod, and then each risk class
        for cp in cp_list:
            print('..Counterparty: ' + cp)
            for prod in self.prod_list:
                df_cp_prod = self.input[(self.input['product_class'] == prod) & (self.input['counterparty_id'] == cp)]
                if len(df_cp_prod) > 0: 
                    prod_margin = self.aggregation(cp,prod,df_cp_prod)
                    Margin_prod[cp][prod] = math.sqrt(np.dot(np.dot(np.transpose(prod_margin),self.CORR_r),prod_margin))
                    self.df_output.loc[self.df_output.shape[0]] = [cp,prod,'All Risk Type',0,0,0,0,Margin_prod[cp][prod]]
            Margin_cp[cp] = sum(list(Margin_prod[cp].values()))
            self.df_output.loc[self.df_output.shape[0]] = [cp,'All Product','All Risk Type',0,0,0,0,Margin_cp[cp]]

        ## output to csv
        outputfile = os.path.join(self.output_path,self.date + '_SIMM_Summary.csv')
        self.df_output.to_csv(outputfile,index=False)

    # aggregation across risk types for one cp and one product
    def aggregation(self,cp,prod,df_cp_prod):
        SIMM_prod = [0.0] * 6
        for risktype in self.risktype_list:
            delta,vega,curv,basecorr = self.risktypes_instance[risktype].calculate(df_cp_prod)
            if delta != 0.0 or vega != 0.0 or curv != 0.0 or basecorr != 0.0:
                self.df_output.loc[self.df_output.shape[0]] = [cp,prod,risktype,delta,vega,curv,0,0]
                SIMM_prod[self.risktype_ind[risktype]] = delta + vega + curv + basecorr
        return SIMM_prod

