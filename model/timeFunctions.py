import time
import logging
import constants.scatterPlotConstants as constants

LEVEL = constants.LEVEL
LOG_FILE = constants.LOG_FILE

timeLog = logging.getLogger(' timer:')
timeLog.setLevel(LEVEL)
formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)
timeLog.addHandler(file_handler)

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
    