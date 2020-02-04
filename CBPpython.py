# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Eckert, Fort, Schott, Yang (2019)
#Corresponding Author: mail@fpeckert.me

import pandas as pd
from ast import literal_eval
import sys
import os

'''
This method takes in a preorder traversal of the tree of codes (which could be
NAICS or geographic) and returns a list of dictionaries whose entries
identify the type of code in the node (naics or geo), its code, and its children.
The method takes in three inputs:
1. The tree of codes (which could be NAICS or geographic). The tree should
    be given as a preorder traversal
2. A name for the codes: Either 'naics' or 'geo'
3. A level function that, given a code, determines the level of the code
    in the directed tree (The level of a node in a directed tree is defined
    as 1 + number of edge between the root and the node.).
Outputs dicts -- a list whose entries have the following structure:
tree[index in the preorder traversal] =
{
  'name': 'naics' OR 'geo'
  'code': code,
  'children': a list of indices of its children in the tree
}
'''
def preorderTraversalToTree(preorder_traversal, name, level_function):
    dicts=[]
    # lineage = [0, index-first-parent, index-second-parent, ....]
    lineage = []

    for index in range(len(preorder_traversal)):
        code = preorder_traversal[index]

        # create this code's dictionary
        code_dict = {}
        code_dict['name'] = name
        code_dict[name] = code
        code_dict['children'] = []

        # find its parent and update the lineage
        level = level_function(code)
        while(len(lineage) >= level and len(lineage) > 0):
            lineage.pop()

        if len(lineage) > 0:
            parents_dictionary = dicts[lineage[-1]]
            parents_dictionary['children'].append(index)

        lineage.append(index)
        dicts.append(code_dict)

    return dicts

# Assumes SIC
def findCorrectionsToTypos(big_df, typos, year, ref_file_name):
    industry_codes = newNaicsCodes(ref_file_name, year)
    industry_tree = preorderTraversalToTree(industry_codes, 'sic', sic_level)

    for typo in typos:
        possible_codes = generatePossibleCodes(typo, year, ref_file_name)

        # numbers = numberTimesCodeOccurs(typo, year, 0, 0)
        # print('\nTypo %s appears %s times in the dataset [national, state, county].' % (typo, numbers))

        # for all occurrences of the typo, find codes that will fit the industrial
        # tree of that occurence
        for index, row in big_df[big_df.sic == typo].iterrows():
            st_id = row.fipstate
            co_id = row.fipscty

            neighborhood = big_df[(big_df.fipstate == st_id) & (big_df.fipscty == co_id)]

            for code in possible_codes:
                index_possible_code = industry_codes.index(code)

                for index_parent, row_parent in neighborhood.iterrows():
                    if row_parent.sic in industry_codes:
                        index_in_codes = industry_codes.index(row_parent.sic)
                        if index_possible_code in industry_tree[index_in_codes]['children']:
                            print('Found correct code for fipstate %d, fipscty %d, code %s. (possible) right code %s and parent %s' % (st_id, co_id, typo, code, row_parent.sic))
                            # print(row_parent.naics)

                            year_before = str(int(year) - 1); year_after = str(int(year) + 1)
                            print('At the same location, the code %s appears %s times in previous year datasets and %s times in next year datasets\n' % (code, numberTimesCodeOccurs(code, year_before, st_id, co_id), numberTimesCodeOccurs(code, year_after, st_id, co_id)) )
    print('')

def generatePossibleCodes(code, year, ref_file_name):
    industry_codes = newNaicsCodes(ref_file_name, year)
    possible_codes = []

    chars = list(range(10))

    for index in range(4):
        for c in chars:
            new_code = code[:index] + str(c) + code[index+1:]
            if (new_code in industry_codes) and (new_code not in possible_codes):
                possible_codes.append(new_code)

    new_code = code[:3] + '\\'
    if new_code in industry_codes:
        possible_codes.append(new_code)

    return possible_codes

def checkNoTimesCodeOccurs(code, year, us, st, co):
    numbers = [us[us.sic == code].sic.size, st[st.sic == code].sic.size, co[co.sic == code].sic.size]
    if numbers[0] > 0 and numbers[1] > 0 and numbers[2] > 0:
        return False

    return ((st[st.sic==code].sic.size <= 5) and (co[co.sic==code].sic.size <= 5))

def numberTimesCodeOccurs(code, year, st_id, co_id):
    us = pd.read_csv('cbp' + year + 'us_edit.csv')
    st = pd.read_csv('cbp' + year + 'st_edit.csv')
    co = pd.read_csv('cbp' + year + 'co_edit.csv')

    big_df = merge_dataframes(us, st, co)

    return big_df[(big_df.fipstate == st_id) & (big_df.fipscty == co_id) & (big_df.sic == code)].index.size

    # if int(year) <= 1997:
    #     return ((st[st.sic==code].sic.size <= 5) and (co[co.sic==code].sic.size <= 5))
    # else:
    #     return ((st[st.naics==code].naics.size <= 5) and (co[co.naics==code].naics.size <= 5))

def typos(year, national_file, state_file, county_file, ref_file_name):
    industry_codes = newNaicsCodes(ref_file_name, year)

    industry_codes_in_dataset = []

    if int(year) <= 1997:
        industry_codes_in_dataset = list(national_file.sic.drop_duplicates()) + list(state_file.sic.drop_duplicates()) + list(county_file.sic.drop_duplicates())
    else:
        industry_codes_in_dataset = list(national_file.sic.drop_duplicates()) + list(state_file.naics.drop_duplicates()) + list(county_file.naics.drop_duplicates())

    # find the codes that are in the datasets but not in the ref files
    typos = list(filter(lambda x: x not in industry_codes, industry_codes_in_dataset))

    # remove codes that appear in every dataset (national, state and county)
    typos = list(filter(lambda x: checkNoTimesCodeOccurs(x, year, national_file, state_file, county_file), typos))

    # drop duplicates
    typos = list(set(typos))

    return typos


def sic_level(code):
    if code == '----':
        return 1
    if '-' in code:
        return 2
    if code[3] == '\\':
        return 3
    if code[3] == '/':
        return 3
    if code[2:4] == '00':
        return 3
    if code[3] == '0':
        return 4
    return 5

# The level function for naics
def naics_level(code):
    # all industries
    if code ==  '------':
        return 1

    return sum(a.isdigit() for a in code)

# level function for geo
def geo_level(code):
    # national
    if code[0] == 0:
        return 1

    # state
    if code[1] == 0:
        return 2

    # county
    return 3

def refFileName(year):
    return "ind_ref_"+str(year)+".csv"
  #  return "cbp" + year + "_ind_ref.csv"
    # names = {
    #     1980: 'sic80.txt',
    #     1981: 'sic81.txt',
    #     1982: 'sic82.txt',
    #     1983: 'sic83.txt',
    #     1984: 'sic84.txt',
    #     1985: 'sic85.txt',
    #     1986: 'sic86_87.txt',
    #     1987: 'sic86_87.txt',
    #     1988: 'sic88_97.txt',
    #     1989: 'sic88_97.txt',
    #     1990: 'sic88_97.txt',
    #     1991: 'sic88_97.txt',
    #     1992: 'sic88_97.txt',
    #     1993: 'sic88_97.txt',
    #     1994: 'sic88_97.txt',
    #     1995: 'sic88_97.txt',
    #     1996: 'sic88_97.txt',
    #     1997: 'sic88_97.txt',
    #     1998: 'naics03.txt',
    #     1999: 'naics03.txt',
    #     2000: 'naics03.txt',
    #     2001: 'naics03.txt',
    #     2002: 'naics03.txt',
    #     2003: 'naics2002.txt',
    #     2004: 'naics2002.txt',
    #     2005: 'naics2002.txt',
    #     2006: 'naics2002.txt',
    #     2007: 'naics2002.txt',
    #     2008: 'naics2008.txt',
    #     2009: 'naics2009.txt',
    #     2010: 'naics2010.txt',
    #     2011: 'naics2011.txt',
    #     2012: 'naics2012.txt',
    #     2013: 'naics2012.txt',
    #     2014: 'naics2012.txt',
    #     2015: 'naics2012.txt',
    #     2016: 'naics2012.txt'
    # }
    # return names[int(year)]

def newNaicsCodes(ref_file, year):

    refs = pd.read_csv(ref_file)
    if year <=1997:
        return list(refs.ind)
    else:
        return list(refs.ind)

# produces a list of naics/sic codes that are ordered like a
# preorder tree traversal. takes in the reference file
# industry reference file's NAICS or SIC column are preordered
def naicsCodes(ref_file_name, year, use=''):

    naics_codes = []
    if year <= 1997:
        if year >= 1988:
            with open(ref_file_name, 'r') as f:
                naics_codes = [line.split(None, 1)[0] for line in f]
        elif year >= 1986:
            with open(ref_file_name, 'r') as f:
                naics_codes = [line.split(None, 1)[0] for line in f]
            naics_codes = naics_codes[1:] # the first one is 'SIC', so remove that one
        elif year >= 1980:
            with open(ref_file_name, 'r') as f:
                naics_codes = [line[0:4] for line in f] # first 4 chars are the code
    else:
        if year <= 2011:
            with open(ref_file_name, 'r') as f:
                naics_codes = [line.split(None, 1)[0] for line in f]
            # remove the first element, which is 'NAICS'
            naics_codes = naics_codes[1:]
        else:
            naics_codes = list(pd.read_csv(ref_file_name).NAICS)

    if use != 'typos':
        # some codes are unused. if you add them to the naics codes, then you
        # have KeyError problems later. so compare them with the national_df
        national_df = pd.read_csv('cbp' + str(year) + 'us_edit.csv')
        real_naics_codes = []
        if year <= 1997:
            real_naics_codes = list(national_df['sic'])
        else:
            real_naics_codes = list(national_df['naics'])

        # have to check these codes are actually used in the dataset
        naics_codes = list(filter(lambda x: x in real_naics_codes, naics_codes))

    # drop duplicated codes but keep order
    return sorted(set(naics_codes))

def geoCodes(state_df, county_df):
    # Create a preorder traversal of the geo tree
    # in list geo_codes
    states = state_df.drop_duplicates(['fipstate'])[['fipstate']].values.tolist()
    counties = county_df.drop_duplicates(['fipstate', 'fipscty'])[['fipstate','fipscty']]
    geo_codes = [(0,0)]
    for state in states:
        state = state[0]
        geo_codes.append((state, 0))
        for county in list(counties[counties.fipstate == state].fipscty):
            geo_codes.append((state, county))

    return geo_codes

def merge_dataframes(national_df, state_df, county_df):
    state_df['fipscty'] = 0
    national_df['fipscty'] = 0
    national_df['fipstate'] = 0
    df = pd.concat([national_df,state_df,county_df], sort=True)
    df['geo'] = list(zip(df.fipstate, df.fipscty))
    return df

# This function submits a query to the data frame and returns a pandas series
# entry is a dictionary with 'geo' representing the geo code (fipstate or 0, fipscty or 0)
# and 'naics' representing the naics code
# It chooses the data frame to search (national, state or county)
# based on the length of the geography argument
def read_df(entry, ub, lb):
    geo = entry['geo']
    naics = entry['naics']
    return (ub[geo][naics], lb[geo][naics])

# write updates the database. it takes in
# 1. the element to be updated (which is a python
#   dictionary that includes geo and naics codes for the element)
# 2. bound to be updated
# 3. the new value for the bound
def write_df(entry, bound, new_value, ub, lb):
    geo = entry['geo']
    naics = entry['naics']

    if bound == 'ub':
        ub[geo][naics] = new_value
    elif bound == 'lb':
        lb[geo][naics] = new_value

    return (ub, lb)

# merges two python dictionaries
def merge_dict(x,y): return {**x, **y}

# checks if two lists of pandas dataframes contain equivalent data frames
def equalListDataFrames(list1, list2):
    for index in range(len(list1)):
        if list1[index].equals(list2[index]) == False:
            return False
    return True

# take str of tuple and return the tuple
def strToTuple(tuple_str):
    tuple_list = tuple_str.replace('(',')').replace(')',',').split(',')
    tuple_list = list(filter(lambda x: x!='', tuple_list))
    return (int(tuple_list[0]), int(tuple_list[1]))

def splitBigDataFrame(big_df, year):
    # from 'geo' create 'fipstate' and 'fipscty'
    big_df[['fipstate', 'fipscty']] = pd.DataFrame(big_df['geo'].tolist(), index = big_df.index)
    big_df = big_df.drop(['geo'], axis = 1)
    big_df = big_df[['naics', 'fipstate', 'fipscty', 'ub', 'lb']]

    # pull national df from big df
    us = big_df.loc[(big_df['fipstate'] == 0)]
    # drop unnecessary columns
    us = us[['naics', 'lb', 'ub']]

    # pull state df from big df and merge missing values
    st = big_df.loc[(big_df['fipstate'] != 0) & (big_df['fipscty'] == 0)]

    original_st = pd.read_csv('cbp' + year + 'st_edit.csv')
    original_st = original_st.rename(index=str, columns={'sic': 'naics'})
    original_st['fipscty'] = 0

    st = pd.merge(st, original_st, on=['naics', 'fipstate', 'fipscty'], how='outer').fillna(0)
    # rename columns
    st = st.rename(index=str, columns={"ub_x": "ub", "lb_x": "lb"})
    # change dtype from float to int
    st.ub = st.ub.astype(int)
    st.lb = st.lb.astype(int)
    # drop unnecessary columns like fipscty
    st = st[['fipstate', 'naics', 'lb', 'ub']]
    st = st.sort_values(by=['fipstate'])

    # oull county df from big df and merge missing values
    co = big_df.loc[(big_df['fipstate'] != 0) & (big_df['fipscty'] != 0)]

    original_co = pd.read_csv('cbp' + year + 'co_edit.csv')
    original_co = original_co.rename(index=str, columns={'sic': 'naics'})

    co = pd.merge(co, original_co, on=['naics', 'fipstate', 'fipscty'], how='outer').fillna(0)
    # rename columns
    co = co.rename(index=str, columns={"ub_x": "ub", "lb_x": "lb"})
    # change datatype from float to int
    co.ub = co.ub.astype(int)
    co.lb = co.lb.astype(int)
    # drop unnecessary columns
    co = co[['fipstate', 'fipscty', 'naics', 'lb', 'ub']]
    co = co.sort_values(by=['fipstate', 'fipscty'])

    return (us, st, co)

def matrixToBigDataFrame(ub, lb):
    ub['naics'] = ub.index
    lb['naics'] = lb.index

    ub_df = pd.melt(ub, id_vars=['naics'], var_name='geo', value_name='ub')
    lb_df = pd.melt(lb, id_vars=['naics'], var_name='geo', value_name='lb')

    df = pd.merge(ub_df, lb_df, on=['naics', 'geo'])

    # some rows were added to make a matrix
    # but they did not exist in the original database
    df = df[df['ub'] != 0]

    return df

def findNonzeroSlack(ub_slack, lb_slack):
    ub_df = pd.melt(ub_slack, id_vars=['naics'], var_name='geo', value_name='ub')
    lb_df = pd.melt(lb_slack, id_vars=['naics'], var_name='geo', value_name='lb')
    df = pd.merge(ub_df, lb_df, on=['naics', 'geo'])
    # delete nonzero entries
    df = df[(df['ub'] != 0) | (df['lb'] != 0)]

    return df

def save(ub, lb, year="2016", optional_name=""):
    big_df = matrixToBigDataFrame(ub, lb)
    (us, st, co) = splitBigDataFrame(big_df, year)

    if int(year) <= 1997:
        us = us.rename(index=str, columns={'naics': 'sic'})
        co = co.rename(index=str, columns={'naics': 'sic'})
        st = st.rename(index=str, columns={'naics': 'sic'})

    us.to_csv("cbp" + year + "us" + optional_name + ".csv", index=False)
    st.to_csv("cbp" + year + "st" + optional_name + ".csv", index=False)
    co.to_csv("cbp" + year + "co" + optional_name + ".csv", index=False)

'''
optimize is a method that takes in a 'fixed location' (which could be a geographical
location like a county or a NAICS code) and a 'variable' tree.
It goes over the tree and optimizes the corresponding entries based on the child-parent
relations in the tree.
'''
def optimize(ub_matrix, lb_matrix, geo_tree, naics_tree,
    location, tree, direction='up', method='children', suppress_output=True):
    if suppress_output == False:
        print('Optimizing. Method: ' + method)

    # direction of the optimization
    r = range(len(tree))
    if direction == 'down': r = reversed(r)

    for index in r:
        node = tree[index]

        # if there is theoretically no children,
        # there is no optimization to be done
        if len(node['children']) == 0:
            continue

        code_upper, code_lower = read_df(merge_dict(node, location), ub_matrix, lb_matrix)

        sum_children_upper, sum_children_lower = (0, 0)
        for c in node['children']:
            (ub,lb) = read_df(merge_dict(tree[c], location), ub_matrix, lb_matrix)
            sum_children_upper += ub
            sum_children_lower += lb

        # if there is no children in the database (even though theoretically
        # there could be), then you can't optimize
        if (sum_children_upper, sum_children_lower) == (0,0):
            continue

        if method == 'children':
            # if none of the children is suppressed, don't update them
            if sum_children_lower == sum_children_upper:
                continue

            for c in node['children']:
                c_upper, c_lower = read_df(merge_dict(tree[c], location), ub_matrix, lb_matrix)

                new_value_upper = min(c_upper, code_upper-(sum_children_lower-c_lower))
                new_value_lower = max(c_lower, code_lower-(sum_children_upper-c_upper))

                # sum of children should be updated
                sum_children_lower += new_value_lower - c_lower
                sum_children_upper += new_value_upper - c_upper

                if (not suppress_output) and ((c_upper, c_lower) != (new_value_upper, new_value_lower)):
                    print(index, c, c_upper, c_lower, new_value_upper, new_value_lower)

                (ub_matrix, lb_matrix) = write_df(merge_dict(tree[c], location), 'ub', new_value_upper, ub_matrix, lb_matrix)
                (ub_matrix, lb_matrix) = write_df(merge_dict(tree[c], location), 'lb', new_value_lower, ub_matrix, lb_matrix)

        elif method == 'parent':
            # if the parent is not suppressed, don't update it
            if code_upper == code_lower:
                continue

            if (not suppress_output) and (sum_children_lower > code_lower or sum_children_upper < code_upper):
                print(index, code_upper, code_lower, sum_children_upper, sum_children_lower)

            new_value_upper = min(sum_children_upper, code_upper)
            new_value_lower = max(sum_children_lower, code_lower)

            # a discrepancy in the data means that there is no overlap between
            # the entry's interval and the interval obtained using its children's
            # sum. The if statement below checks if there is a discrepancy at this entry.
            # if there is a discrepancy in data, print out and exit.
            if max(sum_children_lower, code_lower)>min(sum_children_upper, code_upper):
                print('discrepancy')
                print('index: ' + str(index))
                print('location: ' + str(location))
                print('children sum (lower, upper): ' + str((sum_children_lower, sum_children_upper)))
                print('code (lower, upper): ' + str((code_lower, code_upper)))

                save(ub, lb, optional_name='_problem')

                exit()

            (ub_matrix, lb_matrix) = write_df(merge_dict(node, location),'lb', new_value_lower, ub_matrix, lb_matrix)
            (ub_matrix, lb_matrix) = write_df(merge_dict(node, location),'ub', new_value_upper, ub_matrix, lb_matrix)

    return (ub_matrix, lb_matrix)

# establishment bounds
def fix(ub_matrix, lb_matrix, ub_est, lb_est, geo_tree, naics_tree, suppress_output):
    for geo_index, geo in enumerate(geo_tree):
        for naics_index, naics in enumerate(naics_tree):
            current_entry = merge_dict(naics, geo)

            current_upper, current_lower = read_df(current_entry, ub_matrix, lb_matrix)
            current_upper_est, current_lower_est = read_df(current_entry, ub_est, lb_est)
            est_violates_adding_up_constraints = False

            # NAICS
            # check theoretical children
            if len(naics_tree[naics_index]['children']) != 0:
                sum_children_upper, sum_children_lower = (0,0)

                for child in naics_tree[naics_index]['children']:
                    child_upper, child_lower = read_df(merge_dict(naics_tree[child], geo_tree[geo_index]), ub_matrix, lb_matrix)

                    sum_children_upper += child_upper
                    sum_children_lower += child_lower

                # are there actual children in the dataset
                if (sum_children_upper, sum_children_lower) != (0,0):
                    # is the establishment dataset: (1) better (2) violating adding up constraints?

                    # the establishment dataset is better
                    if current_upper_est < current_upper or current_lower < current_lower_est:

                        # the establishment estimate does not violate adding up constraints
                        if max(current_lower_est, sum_children_lower) <= min(current_upper_est, sum_children_upper):
                            ub_matrix, lb_matrix = write_df(current_entry, 'ub', current_upper_est, ub_matrix, lb_matrix)
                            ub_matrix, lb_matrix = write_df(current_entry, 'lb', current_lower_est, ub_matrix, lb_matrix)
                        else:
                            est_violates_adding_up_constraints = True

            # GEO
            # check theoretical children
            if len(geo_tree[geo_index]['children']) != 0:
                sum_children_upper, sum_children_lower = (0,0)

                for child in geo_tree[geo_index]['children']:
                    child_upper, child_lower = read_df(merge_dict(naics_tree[naics_index], geo_tree[child]), ub_matrix, lb_matrix)

                    sum_children_upper += child_upper
                    sum_children_lower += child_lower

                # are there actual children in the dataset
                if (sum_children_upper, sum_children_lower) != (0,0):
                    # is the establishment dataset: (1) better (2) violating adding up constraints?

                    # the establishment dataset is better
                    if current_upper_est < current_upper or current_lower < current_lower_est:

                        # the establishment estimate does not violate adding up constraints
                        if max(current_lower_est, sum_children_lower) <= min(current_upper_est, sum_children_upper) and (not est_violates_adding_up_constraints):
                            ub_matrix, lb_matrix = write_df(current_entry, 'ub', current_upper_est, ub_matrix, lb_matrix)
                            ub_matrix, lb_matrix = write_df(current_entry, 'lb', current_lower_est, ub_matrix, lb_matrix)

    print('Fixed the dataset.')
    ub_matrix.to_csv('ub_fixed.csv')
    lb_matrix.to_csv('lb_fixed.csv')

    return (ub_matrix, lb_matrix)

# BOUND-TIGHTENING BEGINS HERE
def tighten_bounds(ub_matrix, lb_matrix, geo_tree, naics_tree, year = '16', suppress_output=True):
    print('tightening started')

    while True:
        old_dfs = list(map(lambda x: x.copy(), [ub_matrix, lb_matrix]))

        # STEP 1
        for geo in geo_tree:
            (ub_matrix, lb_matrix) = optimize(ub_matrix, lb_matrix, geo_tree, naics_tree, geo, naics_tree, 'down', 'children', suppress_output)

        # STEP 2
        for naics in naics_tree:
            (ub_matrix, lb_matrix) = optimize(ub_matrix, lb_matrix, geo_tree, naics_tree, naics, geo_tree, 'down', 'children', suppress_output)

        # STEP 3
        for geo in geo_tree:
            (ub_matrix, lb_matrix) = optimize(ub_matrix, lb_matrix, geo_tree, naics_tree, geo, naics_tree, 'up', 'parent', suppress_output)

        # STEP 4
        for naics in naics_tree:
            (ub_matrix, lb_matrix) = optimize(ub_matrix, lb_matrix, geo_tree, naics_tree, naics, geo_tree, 'up', 'parent', suppress_output)

        # check if we're converged
        new_dfs = [ub_matrix, lb_matrix]
        if equalListDataFrames(new_dfs, old_dfs):
            # write data
            ub_matrix.to_csv('ub_converged.csv')
            lb_matrix.to_csv('lb_converged.csv')

            save(ub_matrix, lb_matrix, year, '_tightened_bounds')

            print('converged')
            break
        else:
            ub_matrix.to_csv('ub.csv')
            lb_matrix.to_csv('lb.csv')

            print('no convergence')
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
      
        
        
        
        
        
 #Eckert, Fort, Schott, Yang (2019)
#Corresponding Author: mail@fpeckert.me

import numpy as np, pandas as pd
import re, sys

#Clean data and prepare bounds
def cbp_clean(data_input, geo):

    data_input.columns = map(str.lower, data_input.columns)
    data_input['empflag'] = data_input['empflag'].astype(str)
    data_input.loc[data_input.empflag == ".", 'empflag'] = ""

    if 'lfo' in data_input.columns:
        data_input = data_input[data_input.lfo == '-']
    if 'naics' in data_input.columns:
        data_input = data_input.rename(columns={"naics": "ind"})
    elif 'sic' in data_input.columns:
        data_input = data_input.rename(columns={"sic": "ind"})

    data_input['lb'] = data_input['empflag']
    data_input['ub'] = data_input['empflag']

    data_input['lb'] = data_input['lb'].replace({'A':0,'B':20,'C':100,'E':250,'F':500,'G':1000,'H':2500,'I':5000,'J':10000,'K':25000,'L':50000,'M':100000})
    data_input['ub'] = data_input['ub'].replace({'A':19,'B':99,'C':249,'E':499,'F':999,'G':2499,'H':4999,'I':9999,'J':24999,'K':49999,'L':99999,'M':100000000})

    data_input[['lb','ub']] = data_input[['lb','ub']].apply(pd.to_numeric, errors='coerce')

    data_input.loc[np.isnan(data_input.lb), {'ub','lb'}] = data_input.loc[np.isnan(data_input.lb), 'emp']

    if geo == "us":
        data_input = data_input[['ind','lb','ub']]
    elif geo == "st":
        data_input = data_input[['fipstate','ind','lb','ub']]
    elif geo == "co":
        data_input = data_input[['fipstate','fipscty','ind','lb','ub']]

    return data_input


def cbp_change_code(data_input, code_old, code_new):

    data_helper = data_input.copy()
    data_helper = data_helper[data_helper.ind == code_old]
    data_helper.loc[data_helper.ind == code_old, 'ind'] = code_new
    data_helper.loc[:, 'lb'] = 0
    data_input = data_input.append(data_helper, ignore_index=True)

    return data_input

#Clean data and prepare bounds
def cbp_drop(data_input, year, geo,codechanger):

    ind_drop_set = []
    geo_drop_set = [98,99]

    if geo == "co" and year in range(1970, 1990):
        if year == 1977:
            ind_drop_set = ["0785", "2031", "2433", "2442", "3611", "3716", "3791", "3803","3821", "5122", "5129", "6798", "7012", "7013", "8310", "8361"]

        if year == 1978:
            ind_drop_set = ["0785", "2015", "2433", "2442", "2661", "3611", "3716", "3791", "3803", "3821", "4582", "5129", "6070", "6798", "7012", "7013", "8062", "8084", "8310", "8361", "8411", "8800", "8810"]

        if year == 1979:
            ind_drop_set = ["0785", "0759", "0785", "079/", "1625", "2036", "2940", "2942", "3073", "3239", "3481", "5212", "5513", "5780", "5781", "5820", "5821", "5991", "6122", "6406", "7012", "7060", "7065", "7328", "7380", "7388", "7626", "7638", "7835", "7912",  "7994", "8120", "8126", "8500", "8560", "8562", "8680", "8800", "8810","8811", "3716", "5129", "5192", "6590", "6599", "6798", "7013", "8310", "8361"]

            data_helper = data_input.copy()
            data_helper = data_helper[data_helper.fipstate == 11]
            data_helper = data_helper[data_helper.fipscty == 99]
            data_helper.loc[:, 'lb'] = 0
            data_helper.loc[:, 'fipscty'] = 1
            data_input.append(data_helper, ignore_index=True)

        if year == 1980:
            ind_drop_set = [ "1629", "64--", "6798", "8310", "8361", "8631"]

        if year == 1981:
            ind_drop_set = ["1540", "1542", "6798", "8051", "8310", "8361"]

        if year == 1982:
            ind_drop_set = ["1321", "2771", "8800", "8810", "8811", "3716", "6590", "6599", "6798", "8310", "8361"]

        if year == 1983:
            ind_drop_set = ["1711", "3716", "4229", "6590", "6599", "6793", "6798", "8310", "8361"]

        if year == 1984:
            ind_drop_set = ["3716", "4229", "5380", "5580", "6590", "6599", "6798", "8310", "8361"]

        if year == 1985:
            ind_drop_set = ["3716", "5380", "5580", "6798", "8310", "8361"]

        if year == 1986:
            ind_drop_set = ["1111", "1481", "1531", "1611", "4131", "4151", "4231", "4411", "4431", "4441", "4712", "4811", "4821", "4899", "4911", "4941", "4961", "4971", "5380", "5580", "5970", "6410", "6610", "7840", "8110", "8361"]

        if year == 1987:
            ind_drop_set = ["1540", "4214", "5399", "6410", "8320", "8321", "8330", "8331", "8350", "8351", "8390", "8399"]

        if year == 1988:
            ind_drop_set = ["5399"]

    elif geo == "st" and year in range(70, 91):

        if year == 1977:
            ind_drop_set = ["0785", "2031", "2433", "2442", "3611", "3716", "3791", "3803", "3821", "5129", "6798", "7012", "7013", "8310", "8361"]

        if year == 1978:
            ind_drop_set = ["3716", "5129", "6070", "6798", "7012", "7013", "8310", "8361", "0785", "2015", "2433", "2442", "3611", "3791", "3803", "3821"]

        if year == 1979:
            ind_drop_set = ["0759", "0785", "079/", "1625", "2036", "2940", "2942", "3073", "3239", "3481", "3716", "5129", "5192", "5212", "5513", "5780", "5781", "5820", "5821", "5991", "6406", "6590", "6599", "6798", "7012", "7013", "7060", "7065", "7328", "7380", "7388", "7626", "7638", "7835", "7912", "7994", "8120", "8126", "8310", "8361", "8500", "8560", "8562", "8680", "8800", "8810", "8811"]

        if year == 1980:
            ind_drop_set = ["3716", "6798", "8310", "8361"]

        if year == 1981:
            ind_drop_set = ["1540", "1542", "6798", "8310", "8361"]

        if year == 1982:
            ind_drop_set = ["2771", "3716", "6590", "6599", "6798", "8310", "8361", "8800", "8810", "8811"]

        if year == 1983:
            ind_drop_set = ["3716", "4229", "6590", "6599", "6793", "6798", "8310", "8361"]

        if year == 1984:
            ind_drop_set = ["3716", "4229", "5380", "5580", "6590", "6599", "6798", "8310", "8361"]

        if year == 1985:
            ind_drop_set = ["3716", "5380", "5580", "6798", "8310", "8361"]

        if year == 1986:
            ind_drop_set = ["1111", "1481", "1531", "1611", "4131", "4151", "4231", "4411", "4431", "4441", "4712", "4811", "4821", "4899", "4911", "4941", "4961", "4971", "5380", "5580", "5970", "6410", "6610", "7840", "8110", "8361"]

        if year == 1987:
            ind_drop_set = ["1540", "4214", "6410", "8320", "8330", "8350", "8390"]

        if year == 1988:
            ind_drop_set = ["5399"]

        if year == 1990:
            ind_drop_set = ["8990"]

    elif geo == "us" and year in range(1970, 1990):

        if year == 1977:
            ind_drop_set = ["3716", "5129", "6798", "7012", "7013", "8310", "8361"]
            data_input = codechanger(data_input, "40--", "4300")
            data_input = codechanger(data_input, "40--", "4310")
            data_input = codechanger(data_input, "40--", "4311")

        if year == 1978:
            ind_drop_set = ["3716", "5129", "6070", "6798", "7012", "7013", "8310", "8361"]

        if year == 1979:
            ind_drop_set = ["5129", "6590", "6599", "6798", "7013", "8310", "8361"]
            data_input = codechanger(data_input, "1090", "1092")
            data_input = codechanger(data_input, "6110", "6113")

        if year == 1980:
            ind_drop_set = ["3761", "6798", "8310", "8361"]
            data_input = codechanger(data_input, "1090", "1092")
            data_input = codechanger(data_input, "6110", "6113")

        if year == 1981:
            ind_drop_set = ["6798", "8310", "8361"]
            data_input = codechanger(data_input, "6110", "6113")

        if year == 1982:
            ind_drop_set = ["3716", "6590", "6599", "6798", "8310", "8361"]
            data_input = codechanger(data_input, "1090", "1092")
            data_input = codechanger(data_input, "6110", "6113")

        if year == 1983:
            ind_drop_set = ["3716", "4229", "6590", "6599", "6798", "8310", "8361"]
            data_input = codechanger(data_input, "3570", "3572")
            data_input = codechanger(data_input, "6110", "6113")

        if year == 1984:
            ind_drop_set = ["3716", "4229", "6590", "6599", "6798", "8310", "8361"]
            data_input = codechanger(data_input, "3570", "3572")
            data_input = codechanger(data_input, "3670", "3673")

        if year == 1985:
            ind_drop_set = ["3716", "6798", "8310", "8361"]
            data_input = codechanger(data_input, "3570", "3572")

        if year == 1986:
            ind_drop_set = ["1111", "1481", "1531", "1611", "4131", "4151", "4231", "4411", "4431", "4441", "4712", "4811", "4821", "4899", "4911", "4941", "4961", "4971", "5380", "5580", "5970", "6410", "6610", "7840", "8110", "8361"]
            data_input = codechanger(data_input, "3570", "3572")

        if year == 1987:
            ind_drop_set = ["1110", "1210", "1540", "5399", "6410", "8320", "8321", "8330", "8331", "8350", "8351", "8390", "8399"]
            data_input = codechanger(data_input, "1112", "1110")
            data_input = codechanger(data_input, "1211", "1210")
            data_input = codechanger(data_input, "5800", "5810")


    data_input = data_input[~data_input.ind.isin(ind_drop_set)]

    if geo == 'co' or geo == 'st':
        data_input = data_input[~data_input.fipstate.isin(geo_drop_set)]

    data_input.loc[data_input.ind == "19--", 'ind'] = "20--"
    data_input.loc[data_input.ind == "--", "ind"] = "07--"


    if year == 1997:
        data_input = codechanger(data_input, "5800", "5810")
        data_input = codechanger(data_input, "2070", "2067")

    if year in range(1991, 1997):
        data_input = codechanger(data_input, "5800", "5810")

    if year in range(1970, 1998):
        data_input.ind = data_input.ind.str.replace('/','\\')

    return data_input


##Industry Code Files
def indreforder_ind(data_input):

    data_input = data_input[['NAICS']]
    data_input = data_input.sort_values(by = 'NAICS')

    return data_input


##Geography Code Files
def indreforder_geo(data_input):

    data_input = data_input[['fipstate','fipscty']]
    data_input = data_input.sort_values(by = ['fipstate','fipscty'])

    return data_input       
        
        














import numpy as np, pandas as pd
import re, sys
import fnmatch
import os



##Code to prepare CBP data
geolist = ['co','st','us']

# Using for loop
for year in range(1990,1992):
    for geo in geolist:

        yl = 'cbp'+str(year)+geo

        data = pd.read_csv("Downloads/"+'efsy_cbp_raw_'+str(year)+'/'+geo+'/'+yl+'.txt')

        data = cbp_clean(data,geo)

        data = cbp_drop(data, year, geo, cbp_change_code)

        data.to_csv("Downloads/"+'efsy_cbp_raw_'+str(year)+'/'+geo+'/'+yl+'_edit.csv',index=False)

        print(str(year)+':'+geo+'--done!')



##Code to prepare industry and geo reference files
for year in range(1990, 1992):
    # os.chdir("Downloads/"+'efsy_cbp_raw_'+str(year)+'/ref')
    data = []
    data1 = []
    data2 = []

    for file in os.listdir('.'):
        if fnmatch.fnmatchcase(file, '*sic*'):
            with open (file, 'rt') as myfile:  # Open file lorem.txt for reading text
                for myline in myfile:                 # For each line, read it to a string
                    data.append(str(myline[0:4]))

                df = pd.DataFrame(data, columns=['ind'])
                #df = df.drop(df.index[0])
                df = df.replace('"','')
                df = df.replace("  ","")
                #df.to_csv("Downloads/"+'efsy_cbp_raw_'+str(year)+'/ref/ind_ref_'+str(year)+'.csv', sep='\t',index=False)
                print(df)

        elif fnmatch.fnmatchcase(file, '*naics*'):
            with open (file, 'rt') as myfile:  # Open file lorem.txt for reading text
                for myline in myfile:                 # For each line, read it to a string
                    data.append(str(myline[0:6]))

                df = pd.DataFrame(data, columns=['ind'])
                df = df.replace('"','', regex=True)
                df = df.replace(' ','', regex=True)
                df = df[df.ind != 'NAICS']
                #df.to_csv("Downloads/"+'efsy_cbp_raw_'+str(year)+'/ref/ind_ref_'+str(year)+'.csv', sep='\t',index=False)
                print(df)

        elif fnmatch.fnmatchcase(file, '*geo*'):
            with open (file, 'rt') as myfile:  # Open file lorem.txt for reading text
                for myline in myfile:                 # For each line, read it to a string
                    data1.append(str(myline[1:3]))
                    data2.append(str(myline[6:9]))

                df1 = pd.DataFrame(data1, columns=['fipstate'])
                df2 = pd.DataFrame(data2, columns=['fipstate'])
                df = pd.concat([df1, df2], axis=1)
                df = df.replace('"','', regex=True)
                df = df.replace(' ','', regex=True)
                df = df.replace(',','', regex=True)
                df = df.drop(df.index[0])
                print(df)
                #df.to_csv("Downloads/"+'efsy_cbp_raw_'+str(year)+'/ref/geo_ref_'+str(year)+'.csv', sep='\t',index=False)

        
        
        
    
























#Eckert, Fort, Schott, Yang (2019)
#Corresponding Author: mail@fpeckert.me

##Load Packages
#from gurobipy import *
from gurobipy import *
import cbp
import numpy as np, pandas as pd
import re, sys


model = Model('cbp')

# extract year from the arguments
year = 2016

is_estab = False

if len(sys.argv) > 1:
    year = sys.argv[1]
    if len(sys.argv) > 2:
        is_estab = sys.argv[2] == 'estab'

is_sic = False
if year <= 1997:
    is_sic = True

#Reading in a year's data
national_df = pd.read_csv(root+'/Data Process/'+str(year)+'/us/'+'cbp'+str(year)+'us_edit.csv')
state_df    = pd.read_csv(root+'/Data Process/'+str(year)+'/st/'+'cbp'+str(year)+'st_edit.csv')
county_df   = pd.read_csv(root+'/Data Process/'+str(year)+'/co/'+'cbp'+str(year)+'co_edit.csv')

#rename industry column from sic to naics in sic years.
if is_sic:
    national_df = national_df.rename(index=str, columns={'sic': 'naics'})
    state_df    = state_df.rename(index=str, columns={'sic': 'naics'})
    county_df   = county_df.rename(index=str, columns={'sic': 'naics'})

# find the ref files
industry_ref_file = cbp.refFileName(year)

refpath = root+'/Data Process/'+str(year)+'/ref/'
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



# matrices
ub_matrix = df.pivot(index='ind', columns='geo', values='ub').fillna(0).astype(int)
lb_matrix = df.pivot(index='ind', columns='geo', values='lb').fillna(0).astype(int)

ub_matrix_estab = ub_matrix.copy()
lb_matrix_estab = lb_matrix.copy()

# geo_codes's entries are tuples, which mess up the indexing
# solution: convert the entries to string and remove the space
# because gurobi doesn't like spaces in variable names
geo_codes_str = list(map(lambda x: x.replace(' ', ''), map(str, geo_codes)))


entries = model.addVars(naics_codes, geo_codes_str, name = "Entries")

# add gurobi variables for differences and absolute differences
diffs = model.addVars(naics_codes, geo_codes_str, lb = (-1) * GRB.INFINITY, name = "Diffs")
abs_diffs = model.addVars(naics_codes, geo_codes_str, name = "Abs_Diffs")

if is_estab:
    ub_matrix_estab = df.pivot(index='naics', columns='geo', values='ub_estab').fillna(0).astype(int)
    lb_matrix_estab = df.pivot(index='naics', columns='geo', values='lb_estab').fillna(0).astype(int)

    # Upper bound
    model.addConstrs((entries[naics, geo] <= ub_matrix_estab[geo_codes[geo_index]][naics] for naics in naics_codes for geo_index, geo in enumerate(geo_codes_str)), "ub")
    # Lower bound
    model.addConstrs((entries[naics, geo] >= lb_matrix_estab[geo_codes[geo_index]][naics] for naics in naics_codes for geo_index, geo in enumerate(geo_codes_str)), "lb")
else:
    # Upper bound
    model.addConstrs((entries[naics, geo] <= ub_matrix[geo_codes[geo_index]][naics] for naics in naics_codes for geo_index, geo in enumerate(geo_codes_str)), "ub")
    # Lower bound
    model.addConstrs((entries[naics, geo] >= lb_matrix[geo_codes[geo_index]][naics] for naics in naics_codes for geo_index, geo in enumerate(geo_codes_str)), "lb")

# define diffs and absolute differences
model.addConstrs((diffs[naics, geo] == (entries[naics, geo] - (ub_matrix[geo_codes[geo_index]][naics] + lb_matrix[geo_codes[geo_index]][naics]) / 2.0) for naics in naics_codes for geo_index, geo in enumerate(geo_codes_str)), "difference")
model.addConstrs((abs_diffs[naics, geo] == abs_(diffs[naics, geo]) for naics in naics_codes for geo in geo_codes_str), "absolute_difference")

for geo_index, geo in enumerate(geo_codes_str):
    # print(geo_index)
    for naics_index, naics in enumerate(naics_codes):
        # bounds = (ub_matrix[geo_codes[geo_index]][naics], lb_matrix[geo_codes[geo_index]][naics])

        # # Upper bound
        # model.addConstr((entries[naics, geo] <= bounds[0]), "ub" + naics + geo)
        # model.addConstr((entries[naics, geo] >= bounds[1]), "lb" + naics + geo)

        # model.addConstr((diffs[naics, geo] == (entries[naics, geo] - sum(bounds) / 2.0)), "difference" + naics + geo)
        # model.addConstr((abs_diffs[naics, geo] == abs_(diffs[naics, geo])), "absolute_difference" + naics + geo)

        # Geographical constraints
        # if no children, there is no constraint
        if len(geo_tree[geo_index]['children']) > 0:
            # check whether in reality this cell has children
            children_geo_sum_upper = sum(ub_matrix[geo_codes[child]][naics] for child in geo_tree[geo_index]['children'])
            if children_geo_sum_upper > 0:
                model.addConstr(entries[naics, geo] == sum(entries[naics, geo_codes_str[child]] for child in geo_tree[geo_index]['children']), "Geographical_Constraint" + naics + geo)

        # Industry constraints
        # if no children, there is no constraint
        if len(naics_tree[naics_index]['children']) > 0:
            # check whether this cell has children in reality (in the dataset)
            # if children's upper bound sum is nonzero then there is children in the data
            children_naics_sum_upper = sum(ub_matrix[geo_codes[geo_index]][naics_codes[child]] for child in naics_tree[naics_index]['children'])
            if children_naics_sum_upper > 0:
                # SIC does not have exact hierarchy after level 2. NAICS always has exact hierarchy
                if is_sic and (cbp.sic_level(naics) >= 2):
                    model.addConstr(entries[naics, geo] >= sum(entries[naics_codes[child], geo] for child in naics_tree[naics_index]['children']),  "Industry_Constraint" + naics + geo)
                else:
                    model.addConstr(entries[naics, geo] == sum(entries[naics_codes[child], geo] for child in naics_tree[naics_index]['children']),  "Industry_Constraint" + naics + geo)

# Objective
# obj = entries.sum()
obj = abs_diffs.sum()

# model.setObjective(obj, GRB.MAXIMIZE) # maximize
model.setObjective(obj, GRB.MINIMIZE) # minimize
print('Model created.')


# make the model less sensitive
model.Params.NumericFocus = 1


# model.write("model.lp")

m = model.optimize()

# Write solution to the python variables
for v in model.getVars():
    if v.Varname.split('[')[0] == 'Entries':
        # get naics and geo codes from the variable name
        s = v.Varname.replace(']', '[').split('[')[1]
        naics = s.split(',', 1)[0]
        s = s.split(',', 1)[1]
        geo = tuple(map(int, re.findall('\d+', s)))

        if is_estab:
            # update the matrix
            ub_matrix_estab[geo][naics] = v.X
            lb_matrix_estab[geo][naics] = v.X
        else:
            # update the matrix
            ub_matrix[geo][naics] = v.X
            lb_matrix[geo][naics] = v.X

        # print("%s %f" % (v.Varname, v.X))

# print solution quality statistics
model.printQuality()

# cbp's save function to save the matrices
if is_estab:
    cbp.save(ub_matrix_estab, lb_matrix_estab, year, "_gurobi_midpoint_estab")
else:
    cbp.save(ub_matrix, lb_matrix, year, "_gurobi_midpoint")
    
        
        
        
            
            
            
            