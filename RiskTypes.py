# Based on methodology from ISDA SIMM v2.4

import pandas as pd
import os
import csv
import collections
from collections import defaultdict
import numpy as np
import math
import scipy.stats

# for simplicity, define global variables for single-value parameters
# this is for SIMM v2.4 only
VRW_IR = 0.18
VRW_CreditQ = 0.73
VRW_FX = 0.47
HVR_FX = 0.55
CORR_FX = 0.5

class BaseRiskType:
    def __init__(self):
        self.cor_in_bkt = defaultdict()
        self.RW = defaultdict()
        self.CT = []
        self.cor_across_bkt = [0.0]
        self.VT = defaultdict()
    #abstractmethod
    def initialize(filepath):
        pass
    def calculate(df_input):
        return 0.0,0.0,0.0,0.0
    def aggregateAcrossBucket(self,K2,S,CORR):
        # matrix with different bucket
        margin,values,buckets,n = 0.0,[],[],0
        for bkt_i,s in S.items():
            if bkt_i == 'Residual':
                continue
            margin += K2[bkt_i]
            values.append(s)
            buckets.append(int(bkt_i))
            n += 1
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                matrix[i][j] = values[i] * values[j] * CORR[buckets[i]-1][buckets[j]-1]
        margin += sum(sum(matrix,[]))
        return math.sqrt(margin)
    def scaleFunc(self,t):
        T = 1.0 # avoid error        
        if t == '1m':
            T = 365/12
        elif t == '2w':
            T = 365/26
        elif t == '3m':
            T = 365/4
        elif t == '2y':
            T = 365*2
        elif t == '6m':
            T = 365/2
        elif t == '1y':
            T = 365
        elif t == '3y':
            T = 365*3
        elif t == '5y':
            T = 365*5
        elif t == '10y':
            T = 365*10
        elif t == '15y':
            T = 365*15
        elif t == '20y':
            T = 365*20
        elif t == '30y':
            T = 365*30
        return 0.5 * min(14/T,1)

class IR(BaseRiskType):
    def initialize(self,filepath):
        concentrationthreshold = "Interest Rate risk - Delta Concentration Thresholds.csv"
        riskweight = "Interest Rate risk - Risk weights per vertex (regular currencies).csv"
        rwlow = "Interest Rate risk - Risk weights per vertex (low volatility currencies).csv"
        rwhigh = "Interest Rate risk - Risk weights per vertex (high-volatility currencies).csv"
        equitycorrelation = "Interest Rate Correlation Matrix for Risk Exposures.csv"
        vegathreshold = "Interest Rate risk - Vega Concentration Thresholds.csv"
        CT_file = os.path.join(filepath,concentrationthreshold)
        ## concentration threshold
        CT = []
        with open(CT_file, mode='r', encoding='utf-8-sig') as c:
            reader=csv.DictReader(c)
            for row in reader:
                CT.append(1000000 * float(row['Concentration threshold (USD mm or bp)']))
        self.CT = CT

        RW_file = os.path.join(filepath,riskweight)
        RWlow_file = os.path.join(filepath,rwlow)
        RWhigh_file = os.path.join(filepath,rwhigh)
        CORR_file = os.path.join(filepath,equitycorrelation)
        VT_file = os.path.join(filepath,vegathreshold)
        # across bucket
        CORR_t = np.loadtxt(open(CORR_file, "rb"), delimiter=",")
        self.cor_in_bkt = CORR_t
        self.cor_across_bkt = self.cor_in_bkt
        # risk weight
        RW = defaultdict(list)
        r = pd.read_csv(RW_file)
        for i in range(12):
            RW['Regular'].append(r.iat[0,i])
        l = pd.read_csv(RWlow_file)
        for j in range(12):
            RW['Low'].append(l.iat[0,j])
        h = pd.read_csv(RWhigh_file)
        for k in range(12):
            RW['High'].append(h.iat[0,k])
        self.RW = RW
        # vega CT
        VC = []
        with open(VT_file, mode='r', encoding='utf-8-sig') as c:
            reader=csv.DictReader(c)
            for row in reader:
                VC.append(1000000 * float(row['Concentration threshold (USD mm)']))
        self.VT = VC
        return
    def calculate(self,df_input):
        ## for IR
        d = {'bucket':'first','amount':'sum'}
        df_raw = df_input.groupby(['qualifier','label1','risk_type'],as_index = False).agg(d)
        df = df_raw[df_raw['risk_type'] == 'Risk_IRCurve']
        dfv = df_raw[df_raw['risk_type'] == 'Risk_IRVol']
        deltamargin,vegamargin,curvmargin = 0.0,0.0,0.0
        if len(df) == 0 and len(dfv) == 0:
            return deltamargin,vegamargin,curvmargin,0.0
        #print('.... Interest Rate')
        if len(df) > 0: # only deltamargin
            #print('..... Delta')
            curr_list = list(df['qualifier'].unique())
            CR = self.sumOverBkt(curr_list,df,self.CT)
            # calculate CR and WS
            df['CR'] = df.apply(lambda row: CR[row['qualifier']], axis=1)        
            df['WS'] = df.apply(lambda row: row['amount'] * row['CR'] * self.RW[self.curr2RW(row['qualifier'])][self.label2Bkt(row['label1'])-1], axis=1)
            
            # K-square and S storage
            S,K2 = defaultdict(float),defaultdict(float)
            
            for bkt in curr_list:
                #if df_temp.empty: continue
                # aggregate within bucket
                dftemp = df[df['qualifier'] == bkt]
                K2[bkt],sum_ws = self.aggInBktIR(dftemp,'CR','WS')
                S[bkt] = max(min(sum_ws,math.sqrt(K2[bkt])),-math.sqrt(K2[bkt]))  
            
            deltamargin = self.aggAcrossBktIR(K2,S,CR)
        if len(dfv) > 0: #only vega and curv
            #print('..... Vega and Curvature')
            curr_list = list(dfv['qualifier'].unique())
            VCR = self.sumOverBkt(curr_list,dfv,self.VT)
            # vega
            VRW = VRW_IR
            dfv['VCR'] = dfv.apply(lambda row: VCR[row['qualifier']], axis=1) 
            dfv['VR'] = dfv.apply(lambda row: row['VCR'] * VRW * row['amount'], axis=1)
            Svega,K2vega = defaultdict(float),defaultdict(float)
            
            # curvature
            dfv['CVR'] = dfv.apply(lambda row: self.scaleFunc(row['label1']) * row['amount'], axis=1)
            Scurv,K2curv = defaultdict(float),defaultdict(float)
            cvrabssum,cvrsum = 0.0,0.0# for non-residual and residual
            for bkt in curr_list:
                #if df_temp.empty: continue
                # aggregate within bucket
                dfvtemp = dfv[dfv['qualifier'] == bkt]
                # vega
                K2vega[bkt],sum_vr = self.aggInBktIR(dfvtemp,'VCR','VR')
                Svega[bkt] = max(min(sum_vr,math.sqrt(K2vega[bkt])),-math.sqrt(K2vega[bkt])) 
                # curvature
                K2curv[bkt],sum_cvr,sum_abscvr = self.aggInBktIRCVR(dfvtemp)
                Scurv[bkt] = max(min(sum_cvr,math.sqrt(K2curv[bkt])),-math.sqrt(K2curv[bkt])) 
                cvrsum += sum_cvr
                cvrabssum += sum_abscvr
            vegamargin = self.aggAcrossBktIR(K2vega, Svega, VCR)
            theta = min(cvrsum / cvrabssum,0)
            lambdas = (scipy.stats.norm.ppf(0.995) ** 2 - 1)*(1+theta)-theta
            #curvmargin = max(lambdas * self.aggAcrossBktIRCVR(K2curv, Scurv) + cvrsum,0) * 1 / ((0.44)**2)
            curvmargin = max(lambdas * self.aggAcrossBktIRCVR(K2curv, Scurv) + cvrsum,0)
        return deltamargin,vegamargin,curvmargin, 0.0

    def sumOverBkt(self,curr_list,df,CT):
        CR = defaultdict(float)
        for curr in curr_list:
            CR[curr] = max(1,(abs(df.loc[df['qualifier'] == curr,'amount'].sum())/CT[self.curr2CT(curr)])**0.5)
        return CR

    def aggInBktIR(self,df,R,S):
        # for IR only
        if len(df) == 0: return 0,0
        n = len(df)
        #cr = df[R].to_numpy()
        ws = df[S].to_numpy()
        lb = df['label1'].to_numpy()
        corr_mat = np.full((n,n),1.0)
        for i in range(n):
            for j in range(i+1,n):
                corr = self.cor_in_bkt[self.label2Bkt(lb[i])-1][self.label2Bkt(lb[j])-1]
                #corr_mat[i][j] = corr * min(cr[i],cr[j]) / max(cr[i],cr[j]) if R == 'VCR' else corr * 0.986
                corr_mat[i][j] = corr
                corr_mat[j][i] = corr_mat[i][j]
        
        return np.dot(np.dot(np.transpose(ws),corr_mat),ws),sum(ws)

    def aggInBktIRCVR(self,df):
        # for IR only
        if len(df) == 0: return 0,0
        n = len(df)
        cr = df['CVR'].to_numpy()
        lb = df['label1'].to_numpy()
        corr_mat = np.full((n,n),1.0)
        for i in range(n):
            for j in range(i+1,n):
                corr_mat[i][j] = self.cor_in_bkt[self.label2Bkt(lb[i])-1][self.label2Bkt(lb[j])-1] ** 2
                corr_mat[j][i] = corr_mat[i][j]
        
        return np.dot(np.dot(np.transpose(cr),corr_mat),cr),sum(cr),sum([abs(x) for x in cr])

    def aggAcrossBktIR(self,K2,S,CR):
        # for IR delta only
        deltamargin,values,n,cr = 0.0,[],0,[]
        for bkt_i,s in S.items():
            deltamargin += K2[bkt_i]
            cr.append(CR[bkt_i])
            values.append(s)
            n += 1
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = values[i] * values[j] * min(cr[i],cr[j]) / max(cr[i],cr[j]) * 0.22
        deltamargin += sum(sum(matrix,[]))
        return math.sqrt(deltamargin)

    def aggAcrossBktIRCVR(self,K2,S):
        # for IR only
        margin,values,n = 0.0,[],0
        for bkt_i,s in S.items():
            margin += K2[bkt_i]
            values.append(s)
            n += 1
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = values[i] * values[j] * 0.22 * 0.22
        margin += sum(sum(matrix,[]))
        return math.sqrt(margin)

    def curr2RW(self,curr):
        if curr == 'JPY':
            return 'Low'
        elif curr in ['USD','EUR','GBP','CHF','AUD','NZD','CAD','SEK','NOK','DKK','HKD','KRW','SGD','TWD']:
            return 'Regular'
        else:
            return 'High'

    def curr2CT(self,curr):
        if curr == 'JPY':
            return 3
        elif curr in ['AUD','CAD','CHF','DKK','HKD','KRW','NOK','NZD','SEK','SGD','TWD']:
            return 2
        elif curr in ['USD','EUR','GBP']:
            return 1
        else:
            return 0

    def label2Bkt(self,label):
        if label == '2w':
            bucket = 1
        elif label == '1m':
            bucket = 2
        elif label == '3m':
            bucket = 3
        elif label == '6m':
            bucket = 4
        elif label == '1y':
            bucket = 5
        elif label == '2y':
            bucket = 6
        elif label == '3y':
            bucket = 7
        elif label == '5y':
            bucket = 8
        elif label == '10y':
            bucket = 9
        elif label == '15y':
            bucket = 10
        elif label == '20y':
            bucket = 11
        elif label == '30y':
            bucket = 12
        return bucket

class CreditQ(BaseRiskType):
    def initialize(self,filepath):
        concentrationthreshold = "Credit Qualifying spread risk - Delta Concentration Thresholds.csv"
        riskweight = "Credit Qualifying - Risk weights for all vertices.csv"
        equitycorrelation = "Correlation Qualifying - Correlations (applying to sensitivity or risk exposure pairs within the same bucket).csv"
        correlationparameters = "Credit Qualifying - Correlations parameters applying to sensitivity or risk exposure pairs across different non-resiudal buckets.csv"
        vegathreshold = "Credit spread risk - Vega Concentration Thresholds.csv"
        CT_file = os.path.join(filepath,concentrationthreshold)
        RW_file = os.path.join(filepath,riskweight)
        CORR_file = os.path.join(filepath,correlationparameters)
        CP_file = os.path.join(filepath,equitycorrelation)
        VT_file = os.path.join(filepath,vegathreshold)
        ## concentration threshold
        CT = defaultdict(float)
        with open(CT_file, mode='r', encoding='utf-8-sig') as c:
            reader=csv.DictReader(c)
            for row in reader:
                if row['Buckets'] == "Residual" or row['Buckets'] == "residual":
                    CT['Residual'] = 1000000 * float(row['Concentration threshold (USD mm/ bp)'])
                else:
                    CT[str(int(row['Buckets']))] = 1000000 * float(row['Concentration threshold (USD mm/ bp)'])
        self.CT = CT
        # within bucket
        CORR_in_bkt = [[0.0] * 2 for _ in range(2) ]
        dfcorr = pd.read_csv(CP_file)
        CORR_in_bkt[0][0] = dfcorr.at[0,'Same issuer/seniority, different vertex or currency']
        CORR_in_bkt[0][1] = dfcorr.at[0,'Different issuer/seniority']
        CORR_in_bkt[1][0] = dfcorr.at[1,'Same issuer/seniority, different vertex or currency']
        CORR_in_bkt[1][1] = dfcorr.at[1,'Different issuer/seniority']
        self.cor_in_bkt = CORR_in_bkt
        # across bucket
        self.cor_across_bkt = np.loadtxt(open(CORR_file, "rb"), delimiter=",")
        # risk weight
        RW = defaultdict(float)
        with open(RW_file, mode='r', encoding='utf-8-sig') as c:
            reader=csv.DictReader(c)
            for row in reader:
                if row['Bucket'] == "Residual":
                    RW[str(row['Bucket'])] = float(row['Risk Weight'])
                else:
                    RW[str(int(row['Bucket']))] = float(row['Risk Weight'])
        self.RW = RW
        # vega CT
        dfvt = pd.read_csv(VT_file)
        VT = dfvt.at[0,'Concentration threshold (USD mm)']        
        self.VT = float(VT) * 1000000
        return
    
    def calculate(self,df_input):
        ## for credit
        d = {'amount':'sum'}
        df_raw = df_input.groupby(['bucket','qualifier','label1','risk_type'],as_index = False).agg(d)
        df = df_raw[df_raw['risk_type'] == 'Risk_CreditQ'] # ignore credit non-qualifying for now since no data
        dfv = df_raw[df_raw['risk_type'] == 'Risk_CreditVol']
        deltamargin,vegamargin,curvmargin = 0.0,0.0,[0.0,0.0]
        #basecorrmargin = 0.0
        if len(df) == 0 and len(dfv) == 0:
            return deltamargin,vegamargin,sum(curvmargin),0.0
        #print('.... Credit Qualifying')
        if len(df) > 0: # only deltamargin
            #print('..... Delta')
            bkt_list = list(df['bucket'].unique())
            
            # calculate CR and WS
            df['CR'] = df.apply(lambda row: max((abs(row['amount'])/self.CT[row['bucket']])**0.5,1), axis=1)        
            df['WS'] = df.apply(lambda row: row['amount'] * row['CR'] * self.RW[row['bucket']], axis=1)
            
            # K-square and S storage
            S,K2 = defaultdict(float),defaultdict(float)
            
            # residual
            res = 0.0
            for bkt in bkt_list:
                #if df_temp.empty: continue
                # aggregate within bucket
                dftemp = df[df['bucket'] == bkt]
                if bkt == 'Residual':
                    # delta
                    K2[bkt],sum_ws = self.aggInBktCredit(dftemp,1,'CR','WS')
                    res = math.sqrt(K2[bkt])
                    S[bkt] = res
                else:
                    # delta
                    K2[bkt],sum_ws = self.aggInBktCredit(dftemp,0,'CR','WS')
                    S[bkt] = max(min(sum_ws,math.sqrt(K2[bkt])),-math.sqrt(K2[bkt]))  
            deltamargin = self.aggregateAcrossBucket(K2,S,self.cor_across_bkt) + res
        if len(dfv) > 0: #only vega and curv
            #print('..... Vega and Curvature')
            bkt_list = list(dfv['bucket'].unique())
            
            # vega
            VRW = VRW_CreditQ
            dfv['VCR'] = dfv.apply(lambda row: max(1,(abs(row['amount'])/self.VT)**0.5), axis=1)
            dfv['VR'] = dfv.apply(lambda row: row['VCR'] * VRW * row['amount'], axis=1)
            Svega,K2vega = defaultdict(float),defaultdict(float)
            
            # curvature
            dfv['CVR'] = dfv.apply(lambda row: self.scaleFunc(row['label1']) * row['amount'], axis=1)
            Scurv,K2curv = defaultdict(float),defaultdict(float)
            
            cvrsum,cvrabssum = [0.0,0.0],[0.0,0.0]# for non-residual and residual
            # residual
            resv,resc = 0.0,0.0
            for bkt in bkt_list:
                #if df_temp.empty: continue
                # aggregate within bucket
                dfvtemp = dfv[dfv['bucket'] == bkt]
                if bkt == 'Residual':
                    # vega
                    K2vega[bkt],sum_vr = self.aggInBktCredit(dfvtemp,1,'VCR','VR')
                    resv = math.sqrt(K2vega[bkt])
                    Svega[bkt] = resv
                    # curvature
                    K2curv[bkt],sum_cvr,sum_abscvr = self.aggInBktCreditCVR(dfvtemp,1,'CVR')
                    resc = math.sqrt(K2curv[bkt])
                    Scurv[bkt] = resc
                    cvrsum[1] += sum_cvr
                    cvrabssum[1] += sum_abscvr
                else: 
                    # vega
                    K2vega[bkt],sum_vr = self.aggInBktCredit(dfvtemp,0,'VCR','VR')
                    Svega[bkt] = max(min(sum_vr,math.sqrt(K2vega[bkt])),-math.sqrt(K2vega[bkt])) 
                    # curvature
                    K2curv[bkt],sum_cvr,sum_abscvr = self.aggInBktCreditCVR(dfvtemp,0,'CVR')
                    Scurv[bkt] = max(min(sum_cvr,math.sqrt(K2curv[bkt])),-math.sqrt(K2curv[bkt])) 
                    cvrsum[0] += sum_cvr
                    cvrabssum[0] += sum_abscvr
            vegamargin = self.aggregateAcrossBucket(K2vega, Svega, self.cor_across_bkt) + resv
            for i in range(2):
                theta = min(cvrsum[i] / cvrabssum[i],0) if cvrabssum[i] != 0 else 0
                lambdas = (scipy.stats.norm.ppf(0.995) ** 2 - 1)*(1+theta)-theta
                curvmargin[i] = max(lambdas * self.aggregateAcrossBucket(K2curv, Scurv, np.multiply(self.cor_across_bkt,self.cor_across_bkt)) + cvrsum[i],0)  
        return deltamargin,vegamargin,sum(curvmargin),0.0 # for now basecorrmargin is 0
    
    def aggInBktCredit(self,dftemp,ind,R,S):
        if len(dftemp) == 0: return 0, 0
        n = len(dftemp)
        cr = dftemp[R].to_numpy()
        ws = dftemp[S].to_numpy()
        q = dftemp['qualifier'].to_numpy()
        corr_mat = np.full((n,n),1.0)
        for i in range(n):
                for j in range(i+1,n):
                    if q[i] == q[j] and ws[i] == ws[j]:
                        corr = 0.0
                    else:
                        corr = self.cor_in_bkt[ind][0] if q[i] == q[j] else self.cor_in_bkt[ind][1]
                    corr_mat[i][j] = corr * min(cr[i],cr[j]) / max(cr[i],cr[j])
                    corr_mat[j][i] = corr_mat[i][j]
        return np.dot(np.dot(np.transpose(ws),corr_mat),ws),sum(ws)

    # for credit curvature
    def aggInBktCreditCVR(self,df,ind,CVR):
        if len(df) == 0: return 0,0
        n = len(df)
        cvr = df[CVR].to_numpy()
        q = df['qualifier'].to_numpy()
        corr_mat = np.full((n,n),1.0)
        for i in range(n):
                for j in range(i+1,n):
                    if q[i] == q[j] and cvr[i] == cvr[j]:
                        corr_mat[i][j] = 0.0
                    else:
                        corr_mat[i][j] = self.cor_in_bkt[ind][0] ** 2 if q[i] == q[j] else self.cor_in_bkt[ind][1] ** 2
                    corr_mat[j][i] = corr_mat[i][j]
        
        return np.dot(np.dot(np.transpose(cvr),corr_mat),cvr),sum(cvr),sum([abs(x) for x in cvr])

class Equity(BaseRiskType):
    def initialize(self,filepath):
        concentrationthreshold = "Equity Delta concentration Thresholds.csv"
        riskweight = "Equity-Risk Weights.csv"
        equitycorrelation = "Equity Correlation.csv"
        vegathreshold = "Equity risk - Vega Concentration Thresholds.csv"
        corrwithinbucket = "Equity Correlations within bucket.csv"
        CT_file = os.path.join(filepath,concentrationthreshold)
        ## concentration threshold
        CT = defaultdict(float)
        with open(CT_file, mode='r', encoding='utf-8-sig') as c:
            reader=csv.DictReader(c)
            for row in reader:
                if row['Buckets'] == "Residual":
                    CT[str(row['Buckets'])] = 1000000 * float(row['Concentration threshold (USD mm/ bp)'])
                else:
                    CT[str(int(row['Buckets']))] = 1000000 * float(row['Concentration threshold (USD mm/ bp)'])
        self.CT = CT
        
        RW_file = os.path.join(filepath,riskweight)
        CORR_file = os.path.join(filepath,equitycorrelation)
        VT_file = os.path.join(filepath,vegathreshold)
        CP_file = os.path.join(filepath,corrwithinbucket)
        # within bucket
        CORR_in_bkt = defaultdict(float)
        with open(CP_file, mode='r', encoding='utf-8-sig') as c:
            reader=csv.DictReader(c)
            for row in reader:
                CORR_in_bkt[str(row['Bucket'])] = float(row['Correlation'])
        self.cor_in_bkt = CORR_in_bkt
        # across bucket
        CORR_t = np.loadtxt(open(CORR_file, "rb"), delimiter=",")
        self.cor_across_bkt = CORR_t
        # risk weight
        RW = defaultdict(float)
        with open(RW_file, mode='r', encoding='utf-8-sig') as c:
            reader=csv.DictReader(c)
            for row in reader:
                if row['Bucket'] == "Residual":
                    RW[str(row['Bucket'])] = float(row['Risk Weight'])
                else:
                    RW[str(int(row['Bucket']))] = float(row['Risk Weight'])
        self.RW = RW
        # vega CT
        VC = defaultdict(float)
        with open(VT_file, mode='r', encoding='utf-8-sig') as c:
            reader=csv.DictReader(c)
            for row in reader:
                if row['Bucket'] == "Residual":
                    VC[str(row['Bucket'])] = 1000000.0 * float(row['Concentration threshold (USD mm)'])
                else:
                    VC[str(int(row['Bucket']))] = 1000000.0 * float(row['Concentration threshold (USD mm)'])
        self.VT = VC
        return
    
    def calculate(self,df_input):
        d = {'risk_type':'first','label1':'first','amount':'sum'}
        df_raw = df_input.groupby(['bucket','qualifier'],as_index = False).agg(d)
        
        df = df_raw[df_raw['risk_type'] == 'Risk_Equity']
        
        deltamargin,vegamargin,curvmargin = 0.0,0.0,[0.0,0.0]
        if len(df) == 0:
            return deltamargin,vegamargin,sum(curvmargin),0.0
        if len(df) > 0:
            bkt_list = list(df['bucket'].unique())
        
            # calculate CR and WS
            df['CR'] = df.apply(lambda row: max((abs(row['amount'])/self.CT[row['bucket']])**0.5,1), axis=1)        
            df['WS'] = df.apply(lambda row: row['amount'] * row['CR'] * self.RW[row['bucket']], axis=1)
            
            # K-square and S storage
            S,K2 = defaultdict(float),defaultdict(float)
            # residual
            res = 0.0
            for bkt in bkt_list:
                #if df_temp.empty: continue
                # aggregate within bucket
                K2[bkt],sum_ws = self.aggregateWithinBucket(df,bkt,'CR','WS')
                if bkt == 'Residual':
                    res = math.sqrt(K2[bkt])
                    S[bkt] = res
                else:
                    S[bkt] = max(min(sum_ws,math.sqrt(K2[bkt])),-math.sqrt(K2[bkt]))  
            deltamargin = self.aggregateAcrossBucket(K2,S,self.cor_across_bkt) + res
        
        # vega and curv not implemented yet since no data

        return deltamargin,vegamargin,sum(curvmargin),0.0

    # construct a matrix to calculate K 
    def aggregateWithinBucket(self,df,bkt,R,S):
        df_temp = df[df['bucket'] == bkt]
        n = len(df_temp)
        cr = df_temp[R].to_numpy()
        ws = df_temp[S].to_numpy()
        corr = self.cor_in_bkt[bkt]
        corr_mat = np.full((n,n),1.0)
        for i in range(n):
                for j in range(i+1,n):
                    corr_mat[i][j] = corr * min(cr[i],cr[j]) / max(cr[i],cr[j])
                    corr_mat[j][i] = corr_mat[i][j]
        
        return np.dot(np.dot(np.transpose(ws),corr_mat),ws),sum(ws)

class Commodity(BaseRiskType):
    def initialize(self,filepath):
        concentrationthreshold = "Commodity Delta Concentration Thresholds.csv"
        riskweight = "Commodity - Risk weights.csv"
        commcorrelation = "comm corr.csv"
        vegathreshold = "Commodities risk - Vega Concetration Thresholds.csv"
        corrwithinbucket = "Commodity Correlations within bucket.csv"
        CT_file = os.path.join(filepath,concentrationthreshold)
        ## concentration threshold
        CT = defaultdict(float)
        with open(CT_file, mode='r', encoding='utf-8-sig') as c:
            reader=csv.DictReader(c)
            for row in reader:
                if row['Bucket'] == "Residual":
                    CT[str(row['Bucket'])] = 1000000 * float(row['Concentration threshold (USD mm/%)'])
                else:
                    CT[str(int(row['Bucket']))] = 1000000 * float(row['Concentration threshold (USD mm/%)'])
        self.CT = CT
        
        RW_file = os.path.join(filepath,riskweight)
        CORR_file = os.path.join(filepath,commcorrelation)
        VT_file = os.path.join(filepath,vegathreshold)
        CP_file = os.path.join(filepath,corrwithinbucket)
        # within bucket
        CORR_in_bkt = defaultdict(float)
        with open(CP_file, mode='r', encoding='utf-8-sig') as c:
            reader=csv.DictReader(c)
            for row in reader:
                CORR_in_bkt[str(row['Bucket'])] = float(row['Correlation'])
        self.cor_in_bkt = CORR_in_bkt
        # across bucket
        CORR_t = np.loadtxt(open(CORR_file, "rb"), delimiter=",")
        self.cor_across_bkt = CORR_t
        # risk weight
        RW = defaultdict(float)
        with open(RW_file, mode='r', encoding='utf-8-sig') as c:
            reader=csv.DictReader(c)
            for row in reader:
                if row['Bucket'] == "Residual":
                    RW[str(row['Bucket'])] = float(row['Risk Weight'])
                else:
                    RW[str(int(row['Bucket']))] = float(row['Risk Weight'])
        self.RW = RW
        # vega CT
        VC = defaultdict(float)
        with open(VT_file, mode='r', encoding='utf-8-sig') as c:
            reader=csv.DictReader(c)
            for row in reader:
                if row['Bucket'] == "Residual":
                    VC[str(row['Bucket'])] = 1000000.0 * float(row['Concentration threshold (USD mm)'])
                else:
                    VC[str(int(row['Bucket']))] = 1000000.0 * float(row['Concentration threshold (USD mm)'])
        self.VT = VC
        return

    def calculate(self,df_input):
        d = {'risk_type':'first','label1':'first','amount':'sum'}
        df_raw = df_input.groupby(['bucket','qualifier'],as_index = False).agg(d)
        
        df = df_raw[df_raw['risk_type'] == 'Risk_Commodity']
        
        deltamargin,vegamargin,curvmargin = 0.0,0.0,[0.0,0.0]
        if len(df) == 0:
            return deltamargin,vegamargin,sum(curvmargin),0.0
        if len(df) > 0:
            bkt_list = list(df['bucket'].unique())
        
            # calculate CR and WS
            df['CR'] = df.apply(lambda row: max((abs(row['amount'])/self.CT[row['bucket']])**0.5,1), axis=1)        
            df['WS'] = df.apply(lambda row: row['amount'] * row['CR'] * self.RW[row['bucket']], axis=1)
            
            # K-square and S storage
            S,K2 = defaultdict(float),defaultdict(float)
            # residual
            res = 0.0
            for bkt in bkt_list:
                # aggregate within bucket
                K2[bkt],sum_ws = self.aggregateWithinBucket(df,bkt,'CR','WS')
                if bkt == 'Residual':
                    res = math.sqrt(K2[bkt])
                    S[bkt] = res
                else:
                    S[bkt] = max(min(sum_ws,math.sqrt(K2[bkt])),-math.sqrt(K2[bkt]))  
            deltamargin = self.aggregateAcrossBucket(K2,S,self.cor_across_bkt) + res
        
        # vega and curv not implemented yet since no data
        return deltamargin,vegamargin,sum(curvmargin),0.0

    # construct a matrix to calculate K 
    def aggregateWithinBucket(self,df,bkt,R,S):
        df_temp = df[df['bucket'] == bkt]
        n = len(df_temp)
        cr = df_temp[R].to_numpy()
        ws = df_temp[S].to_numpy()
        corr = self.cor_in_bkt[bkt]
        corr_mat = np.full((n,n),1.0)
        for i in range(n):
                for j in range(i+1,n):
                    corr_mat[i][j] = corr * min(cr[i],cr[j]) / max(cr[i],cr[j])
                    corr_mat[j][i] = corr_mat[i][j]
        
        return np.dot(np.dot(np.transpose(ws),corr_mat),ws),sum(ws)

class FX(BaseRiskType):
    def initialize(self,filepath):
        concentrationthreshold = "FX risk - Delta Concentration Thresholds.csv"
        CT_file = os.path.join(filepath,concentrationthreshold)
        ## concentration threshold
        CT = []
        with open(CT_file, mode='r', encoding='utf-8-sig') as c:
            reader=csv.DictReader(c)
            for row in reader:
                CT.append(1000000 * float(row['Concentration threshold (USD mm/%)']))
        self.CT = CT
        # corr
        correlation = "FX - Correlations regular.csv"# only reg since data only have CAD now
        CORR_file = os.path.join(filepath,correlation)
        CORR = [[0.0] * 2 for _ in range(2)]
        df = pd.read_csv(CORR_file)
        for i in range(2):
            CORR[i][0] = float(df.at[i,'Regular'])
            CORR[i][1] = float(df.at[i,'High'])
        self.cor_in_bkt = CORR
        self.cor_across_bkt = CORR
        # risk weight
        riskweight = 'Foreign Exchange - Risk weights.csv'
        RW_file = os.path.join(filepath,riskweight)
        RW = [[0.0] * 2 for _ in range(2)]
        df = pd.read_csv(RW_file)
        for i in range(2):
            RW[i][0] = float(df.at[i,'Regular'])
            RW[i][1] = float(df.at[i,'High'])
        self.RW = RW
        # vega CT
        vegathreshold = "FX risk - Vega Concentration Thresholds.csv"
        VT_file = os.path.join(filepath,vegathreshold)
        VT = [[0.0] * 3 for _ in range(3)]
        df = pd.read_csv(VT_file)
        for i in range(3):
            for j in range(3):
                VT[i][j] = float(df.iat[i,j+1]) * 1000000
        self.VT = VT
        return
    
    def calculate(self,df_input):
        df_input = df_input[df_input['qualifier'] != df_input['amount_currency']]
        d = {'amount':'sum','amount_currency':'first'}
        df_raw = df_input.groupby(['qualifier','risk_type'],as_index = False).agg(d)
        df = df_raw[df_raw['risk_type'] == 'Risk_FX']
        dfv = df_input[df_input['risk_type'] == 'Risk_FXVol']
        corr_vol = CORR_FX
        deltamargin,vegamargin,curvmargin = 0.0,0.0,0.0
        if len(df) == 0 and len(dfv) == 0:
            return deltamargin,vegamargin,curvmargin,0.0
        if len(df) > 0:
            # calculate CR and WS
            df['CR'] = df.apply(lambda row: max((abs(row['amount'])/self.CT[self.curr2CTFX(row['qualifier'])])**0.5,1), axis=1)    
            df['WS'] = df.apply(lambda row: row['amount'] * row['CR'] * self.RW[self.curr2RWFX(row['qualifier'])][self.curr2RWFX(row['amount_currency'])], axis=1)
            deltamargin = self.aggFXDelta(df) # hard code because only CAD now:
        if len(dfv) > 0:
            alpha,HVR,VRW = scipy.stats.norm.ppf(0.99),HVR_FX,VRW_FX #hardcode for v2.4
            # vega
            dfv['sigma'] = dfv.apply(lambda row: self.RW[self.curr2RWFX(row['qualifier'])][self.curr2RWFX(row['amount_currency'])] * math.sqrt(365/14) / alpha, axis = 1)
            dfv['VR_i'] = dfv.apply(lambda row: row['sigma'] * row['amount'] * HVR, axis=1)
            # currency code is 3-digi
            dfv['VCR'] = dfv.apply(lambda row: max(1,(abs(row['VR_i'])/self.VT[self.curr2CTFX(row['qualifier'][:3])][self.curr2CTFX(row['qualifier'][3:])])**0.5), axis=1)
            dfv['VR'] = dfv.apply(lambda row: row['VR_i'] * VRW * row['VCR'], axis=1)
            dfv['CVR'] = dfv.apply(lambda row: self.scaleFunc(row['label1']) * row['amount'] * row['sigma'], axis=1)
            d = {'amount':'sum','VR':'sum','CVR':'sum'}
            dff = dfv.groupby(['qualifier'],as_index = False).agg(d)
            vegamargin,cvrsum,cvrabssum, = self.aggFXVol(dff,corr_vol,'VR') 
            # curvature
            curvm,cvrsum,cvrabssum, = self.aggFXVol(dff,corr_vol**2,'CVR') 
            theta = min(cvrsum / cvrabssum,0) if cvrabssum != 0 else 0
            lambdas = (scipy.stats.norm.ppf(0.995) ** 2 - 1)*(1+theta)-theta
            curvmargin = max(curvm * lambdas + cvrsum,0)
        # hard code for vega and curv now since no data
        return deltamargin,vegamargin,curvmargin,0.0
    
    def aggFXDelta(self,df):
        ws = df['WS'].to_numpy()
        q = df['qualifier'].to_numpy()
        n = len(df)
        corr_mat = np.full((n,n),1.0)
        for i in range(n):
            for j in range(i+1,n):
                corr_mat[i][j] = self.cor_in_bkt[self.curr2RWFX(q[i])][self.curr2RWFX(q[j])]
                corr_mat[j][i] = corr_mat[i][j]
        return math.sqrt(np.dot(np.dot(np.transpose(ws),corr_mat),ws))
        
    def aggFXVol(self,df,corr,S):
        k = df[S].to_numpy()
        #vcr = df['VCR'].to_numpy()
        #q = df['qualifier'].to_numpy()
        n = len(df)
        corr_mat = np.full((n,n),1.0)
        for i in range(n):
            for j in range(i+1,n):
                #corr_mat[i][j] = corr * min(vcr[i],vcr[j]) / max(vcr[i],vcr[j]) if S == 'VR' else corr
                corr_mat[i][j] = corr
                corr_mat[j][i] = corr_mat[i][j]
        return math.sqrt(np.dot(np.dot(np.transpose(k),corr_mat),k)),sum(k),sum([abs(x) for x in k])
    
    def curr2CTFX(self,curr):
        if curr in ['USD','EUR','JPY','GBP','AUD','CHF','CAD']:
            return 0
        elif curr in ['BRL','CNY','HKD','INR','KRW','MXN','NOK','NZD','RUB','SEK','SGD','TRY','ZAR']:
            return 1
        else:
            return 2
    
    def curr2RWFX(self,curr):
        if curr in ['ARS','BRL','MXN','TRY','ZAR']:
            return 1
        else:
            return 0