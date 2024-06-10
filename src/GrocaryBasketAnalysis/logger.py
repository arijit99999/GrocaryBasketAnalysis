import os
import logging
from datetime import datetime
log_file=f"{datetime.now().strftime('%d_%m_%y_%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),'logs')
os.makedirs(log_path,exist_ok=True)
log_file_path=os.path.join(log_path,log_file)



logging.basicConfig(level=logging.INFO,filename=log_file_path,
format=('%(lineno)d--%(levelname)s--%(asctime)s--%(message)s'))