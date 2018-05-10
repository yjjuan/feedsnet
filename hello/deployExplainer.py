# -*- coding: utf-8 -*-
"""
Created on Sat May  5 20:10:00 2018

@author: nanobioscience
"""
import lime.lime_tabular
import numpy as np


class explainer:

    def __init__(self, training_data, feature_names,class_names,categorical_features, categorical_names, kernel_width):
        self.training_data = training_data
        self.feature_names = feature_names
        self.class_names = class_names
        self.categorical_features = categorical_features
        self.categorical_names = categorical_names
        self.kernel_width = kernel_width
        self.explainer = lime.lime_tabular.LimeTabularExplainer(training_data ,feature_names = feature_names,class_names=class_names,
                                                           categorical_features=categorical_features, 
                                                           categorical_names=categorical_names, kernel_width=3)
    
    def explain(self, model, case):
        np.random.seed(1)
        i = 30
        user_case = case
        predict_fn = lambda x: model.predict_proba(x).astype(float)
        
        exp = self.explainer.explain_instance(user_case, predict_fn, num_features=5)
      
        final_exp = dict()
        weight_list = exp.__dict__['local_exp'][1]
        for i in range(self.training_data.shape[1]):
            final_exp[i] = dict()
            if i in self.categorical_features:
                original_desc = exp.__dict__['domain_mapper'].__dict__['discretized_feature_names'][i]
                desc_list = original_desc.split('=')
                final_exp[i]['desc'] = ' is '.join(desc_list)
            else:
                # Operation on numerical features
                boundary = np.percentile(self.training_data[:,i],(0,25,50,75,100))
                interval = 0
                if user_case[i] >= boundary[4]:
                    final_exp[i]['desc'] = '{} is very high'.format(self.feature_names[i])
                else:
                    interval = list(user_case[i]<boundary).index(True)
                    
                    if interval == 0:
                        final_exp[i]['desc'] = '{} is very low'.format(self.feature_names[i])
                    elif interval == 1:
                        final_exp[i]['desc'] = '{} is low'.format(self.feature_names[i])
                    elif interval == 2:
                        final_exp[i]['desc'] = '{} is slightly low'.format(self.feature_names[i])
                    elif interval == 3:
                        final_exp[i]['desc'] = '{} is slightly high'.format(self.feature_names[i])
                    elif interval == 4:
                        final_exp[i]['desc'] = '{} is high'.format(self.feature_names[i])
                    else:
                        print('Error')
        
        for idx, weight in weight_list:
            final_exp[idx]['weight'] = weight
               
        #print(final_exp)   
        return final_exp
