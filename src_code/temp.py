import misc
import numpy as np
import relationships

list1 = [1,5,6,7,8,9,9,4,5,6,2,4]
list2 = [2,1,5,6,7,8,9,9,4,5,6,2]


causal_dict = relationships.compute_causality(list1,list2,max_lag=3)
# misc.print_causality_dict(causal_dict)
