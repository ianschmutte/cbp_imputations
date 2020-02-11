# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:35:04 2020

@author: manav
"""

#Eckert, Fort, Schott, Yang (2019)
#Corresponding Author: mail@fpeckert.me

##Load Packages
#from gurobipy import *
os.chdir("C:/Users/manav/Downloads")

from gurobipy import *
import cbpfunc as cbp
import numpy as np, pandas as pd
import re, sys


model = Model('cbp')

# extract year from the arguments
year = 1990

is_estab = False

if len(sys.argv) > 1:
    year = sys.argv[1]
    if len(sys.argv) > 2:
        is_estab = sys.argv[2] == 'estab'

is_sic = False
if year <= 1997:
    is_sic = True

#Reading in a year's data
national_df = pd.read_csv("C:/Users/manav/Downloads/"+'efsy_cbp_raw_'+str(year)+'/us/'+'cbp'+str(year)+'us_edit.csv')
state_df    = pd.read_csv("C:/Users/manav/Downloads/"+'efsy_cbp_raw_'+str(year)+'/st/'+'cbp'+str(year)+'st_edit.csv')
county_df   = pd.read_csv("C:/Users/manav/Downloads/"+'efsy_cbp_raw_'+str(year)+'/co/'+'cbp'+str(year)+'co_edit.csv')

#rename industry column from sic to naics in sic years.
if is_sic:
    national_df = national_df.rename(index=str, columns={'sic': 'naics'})
    state_df    = state_df.rename(index=str, columns={'sic': 'naics'})
    county_df   = county_df.rename(index=str, columns={'sic': 'naics'})

# find the ref files
industry_ref_file = cbp.refFileName(year)

refpath = "C:/Users/manav/Downloads/"+'efsy_cbp_raw_'+str(year)+'/ref/'
os.chdir(refpath)

naics_codes       = cbp.newNaicsCodes(industry_ref_file, year)

geo_codes         = cbp.geoCodes(state_df, county_df)


# ##
# Construct tree for NAICS codes
# ##
# determine level function based on which industry code is used
industry_level_function = cbp.naics_level
if is_sic:
    industry_level_function = cbp.sic_level

naics_tree = cbp.preorderTraversalToTree(naics_codes, 'naics', industry_level_function)

# ##
# Construct tree for Geography
# ##
geo_tree = cbp.preorderTraversalToTree(geo_codes, 'geo', cbp.geo_level)

df = cbp.merge_dataframes(national_df, state_df, county_df)

os.chdir("C:/Users/manav")

df.to_csv("Downloads/"+'efsy_cbp_raw_'+str(year)+'/'+'finalsolved'+str(year)+'.csv',index=False)
