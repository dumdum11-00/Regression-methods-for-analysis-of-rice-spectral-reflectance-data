import pandas as pd
import numpy as np
import os
import re
from pre_proc_nvdi import set_up_data
from linear_regression import linear_regression
from gradient_boost import gradient_boost


red_wl= 625 ## 625-740nm
ir_wl = 750 ## 750-1mm 

# while red_wl <= 740 :
#   while ir_wl <= 2500 :
#     print(red_wl, ir_wl)
#     ir_wl += 1
#   red_wl+= 1
        # set_up_data(red_wl, ir_wl)

for red_wl in range(625, 741):
    for ir_wl in range (750, 2501):
        set_up_data(red_wl, ir_wl)
        # Run stuff         
        gradient_boost(red_wl, ir_wl)
