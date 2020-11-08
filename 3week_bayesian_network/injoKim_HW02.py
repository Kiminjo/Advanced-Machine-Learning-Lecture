# -*- coding: utf-8 -*-

# DO NOT CHANGE
import pandas as pd
import numpy as np

def get_order(structure) :
    ordered_list= list()
    
    for key, value in structure.items():
        if not value :
            ordered_list.append(key)

    for key, value in structure.items():            
        for ele in value :
            if ele in ordered_list and key not in ordered_list :
                ordered_list.append(key)
        
    for key, value in structure.items():        
        for attribute in value:
            if attribute not in ordered_list :
                ordered_list.append(attribute)
                
    for key, value in structure.items():
        if key not in ordered_list :
            ordered_list.append(key)
            
    return ordered_list
"""
def get_order(structure):
    
    def order(df, order_list) :
            
        #dataframe 재정렬
        df.sort_values(by=['value'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # order_list에 key값 저장
        order_var = df['key'][0]
        order_list.append(order_var)
        #저장후 제거
        df.drop([0], axis=0, inplace=True)
        for index, value in df.iterrows() :
            if order_var in df['value'][index] :
                df['value'][index].remove(order_var)
    
        return df, order_list
    
    #make data frame
    key_list = []
    value_list = []
    
    #find total parents
    for key, value in structure.items():
        key_list.append(key)
        value_list.append(value)
        df = pd.DataFrame({'key' : key_list, 'value' : value_list})
        
    iter_num = len(df)
    order_list = []
          
    for length in range(iter_num) : 
        df, order_list= order(df,order_list)
        
    return order_list
"""

def learn_parms(data,structure,var_order):
    # data: training data
    # structure: dictionary of structure
    # var_order: list of learning order of variables
    # return dictionary of trained parameters (key=variable, value=learned parameters)
    
    dict_ = {}
    for var in var_order :
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
parms1=learn_parms(data, str1, get_order(str1))
print('-----First Structure------')
print_parms(order1,parms1)
print('')

str2={'A':['E'],'S':['A','E'],'E':['O','R'],'O':['R','T'],'R':['T'],'T':[]}
order2=get_order(str2)
parms2=learn_parms(data,str2,get_order(str2))
print('-----Second Structure-----')
print_parms(order2,parms2)
