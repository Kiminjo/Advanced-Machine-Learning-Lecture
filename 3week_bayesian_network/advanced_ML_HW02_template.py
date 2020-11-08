# -*- coding: utf-8 -*-

# DO NOT CHANGE
import pandas as pd
import numpy as np

def get_order(structure):
    
    ordered_variable = []
    
    #find total parents
    for key, value in structure.items():
        if len(value) == 0:
            ordered_variable.append(key)
    
    #append child to oredered_variable
    for paraents in ordered_variable:
        for key, value in structure.items() :
            if paraents in value :
                if key in ordered_variable :
                    break ;
                ordered_variable.append(key)
    
    return ordered_variable
    

def learn_parms(data,structure,var_order):
    # data: training data
    # structure: dictionary of structure
    # var_order: list of learning order of variables
    # return dictionary of trained parameters (key=variable, value=learned parameters)
    
    dict_ = {}
    for var in var_order :
        df__ = pd.DataFrame()
        matrix = []
        key_list = []
        
        if len(structure[var]) == 0 :
            df_ =  pd.DataFrame(list(data[var].value_counts(normalize=True))).T
            df_.columns = np.unique(data[var])
            dict_[var] =  df_
            continue ;
        
        grouped = data.groupby(structure[var])
        row_number = 0
        for key, group in grouped :
            matrix.append(list(group[var].value_counts(normalize=True)))
            df = pd.DataFrame(matrix)
            df.columns = np.unique(data[var])
            key_list.append(key)
        
        df.index = key_list
        dict_[var] = df
        
    return dict_
    
                
def print_parms(var_order,parms):
    # var_order: list of learning order of variables
    # parms: dictionary of trained parameters (key=variable, value=learned parameters)
    # print the trained parameters for each variable
    for var in var_order:
        print('-------------------------')
        print('Variable Name=%s'%(var))
        #TODO: print the trained paramters
        
        print(parms[var])
        
        
        
data=pd.read_csv('https://drive.google.com/uc?export=download&id=1taoE9WlUUN4IbzDzHv7mxk_xSj07f-Zt', sep=' ')

str1={'A':[],'S':[],'E':['A','S'],'O':['E'],'R':['E'],'T':['O','R']}
order1=get_order(str1)
parms1=learn_parms(data,str1,get_order(str1))
print('-----First Structure------')
print_parms(order1,parms1)
print('')

str2={'A':['E'],'S':['A','E'],'E':['O','R'],'O':['R','T'],'R':['T'],'T':[]}
order2=get_order(str2)
parms2=learn_parms(data,str2,get_order(str2))
print('-----Second Structure-----')
print_parms(order2,parms2)
print('')