
"""Only to run files."""

import employee
import config as cfg
import processing_all_files
import  networkx as nx
import network
import interactions
import numpy as np

import employee
import misc
import plot

distance_dict = interactions.compute_distance_between_business_and_social_embedding(cfg.SENTIMENT_BUSINESS,cfg.SENTIMENT_PERSONAL,employee.employee_list,
                                                                                    k =5, start_week= 125, end_week= 130, in_network= True, only_week=True)


distance_dict = misc.change_key_string_key_date(distance_dict)
dates_list  = []
distance_list = []

for date in sorted(distance_dict.keys()):
    distance_list.append(distance_dict[date])
    dates_list.append(date)

plot.plot_list_vs_dates(dates_list,distance_list,"","","","")

print(distance_dict)