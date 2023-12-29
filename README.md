# ISDA-SIMM-v2.4-Daily-Reporting
This is a risk engineering tool that's used for daily reporting of initial margin calculated based on ISDA SIMM v2.4 methodology, which is the official methodology for risk-based approach for calculating the initial margin for non-cleared OTC derivatives, incorporating delta risk, vega risk, curvature risk, concentration risk and credit base correlation risk.

The full documentation can be found https://www.isda.org/a/CeggE/ISDA-SIMM-v2.4-PUBLIC.pdf

The tool takes sensitivities source data and produces a comprehensive summary of initial margin calculated based on ISDA SIMM v2.4 methodology. 
The result is hierarchical by counterparty, product and then risk types.

The tool can be run either from the script, or using command line arguments "python main.py --date YYYY-MM-DD"

In the config json file, you can set up:
1. The naming convention for input files
2. SIMM version, for this tool it's "v2.4"
3. which product you would like to include. According to ISDA, there are a total of 4 products: Equity, Credit, Commodity and RatesFX
4. which risk type you would like to include. According to ISDA, there are a total of 6 risk types: IR, Credit Qualifying, Credit Non-Qualifying, Equity, Commodity, FX

The tool will produce a csv file summarizing the hierarchical initial margin results under each counterparty, product and risk type.

Please note that Equity and Commodity non-delta risk, credit base correlation risk, and Credit Non-Qualifying are not implemented since the developer lacks data.
