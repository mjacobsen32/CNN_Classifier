import time
import logging
import constants.scatterPlotConstants as constants
from model.logger import modelLog as timeLog

def timer_wrapper(func):
    total_time = 0
    
    def timer(*args, **kwargs): 
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        elapsed = t1 - t0
        
        # bring variables into scope
        nonlocal total_time
        total_time += elapsed
        
        timeLog.debug(f"{func.__name__} took {elapsed:0.4f} seconds")
            
        
        return result
    return timer
    