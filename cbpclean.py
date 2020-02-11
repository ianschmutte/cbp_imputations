# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:34:33 2020

@author: manav
"""
os.chdir("C:/Users/manav/Downloads")

import numpy as np, pandas as pd
import re, sys
import fnmatch
import os

import cbputils

os.chdir("C:/Users/manav")

##Code to prepare CBP data
geolist = ['co','st','us']

# Using for loop
for year in range(1990,1992):
    for geo in geolist:

        yl = 'cbp'+str(year)+geo

        data = pd.read_csv("Downloads/"+'efsy_cbp_raw_'+str(year)+'/'+geo+'/'+yl+'.txt')

        data = cbputils.cbp_clean(data,geo)

        data = cbputils.cbp_drop(data, year, geo, cbputils.cbp_change_code)

        data.to_csv("Downloads/"+'efsy_cbp_raw_'+str(year)+'/'+geo+'/'+yl+'_edit.csv',index=False)

        print(str(year)+':'+geo+'--done!')



##Code to prepare industry and geo reference files
for year in range(1990, 1992):
    os.chdir("C:/Users/manav/Downloads/"+'efsy_cbp_raw_'+str(year)+'/ref')
    dat = []
    dat1 = []
    dat2 = []

    for file in os.listdir('.'):
        if fnmatch.fnmatchcase(file, '*sic*'):
            with open (file, 'rt') as myfile:  # Open file lorem.txt for reading text
                for myline in myfile:                 # For each line, read it to a string
                    dat.append(str(myline[0:4]))

                df = pd.DataFrame(dat, columns=['ind'])
                #df = df.drop(df.index[0])
                df = df.replace('"','')
                df = df.replace("  ","")
                df.to_csv("C:/Users/manav/Downloads/"+'efsy_cbp_raw_'+str(year)+'/ref/ind_ref_'+str(year)+'.csv', sep='\t',index=False)
                print(df)

        elif fnmatch.fnmatchcase(file, '*naics*'):
            with open (file, 'rt') as myfile:  # Open file lorem.txt for reading text
                for myline in myfile:                 # For each line, read it to a string
                    dat.append(str(myline[0:6]))

                df = pd.DataFrame(dat, columns=['ind'])
                df = df.replace('"','', regex=True)
                df = df.replace(' ','', regex=True)
                df = df[df.ind != 'NAICS']
                df.to_csv("C:/Users/manav/Downloads/"+'efsy_cbp_raw_'+str(year)+'/ref/ind_ref_'+str(year)+'.csv', sep='\t',index=False)
                print(df)

        elif fnmatch.fnmatchcase(file, '*geo*'):
            with open (file, 'rt') as myfile:  # Open file lorem.txt for reading text
                for myline in myfile:                 # For each line, read it to a string
                    dat1.append(str(myline[1:3]))
                    dat2.append(str(myline[6:9]))

                df1 = pd.DataFrame(dat1, columns=['fipstate'])
                df2 = pd.DataFrame(dat2, columns=['fipstate'])
                df = pd.concat([df1, df2], axis=1)
                df = df.replace('"','', regex=True)
                df = df.replace(' ','', regex=True)
                df = df.replace(',','', regex=True)
                df = df.drop(df.index[0])
                print(df)
                df.to_csv("C:/Users/manav/Downloads/"+'efsy_cbp_raw_'+str(year)+'/ref/geo_ref_'+str(year)+'.csv', sep='\t',index=False)
