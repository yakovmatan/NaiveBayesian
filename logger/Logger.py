import logging
import os
from datetime import datetime

log_dir = "../logs"
os.makedirs(log_dir, exist_ok=True)

log_filename = datetime.now().strftime("log_%Y-%m-%d.log")
log_path = os.path.join(log_dir, log_filename)

logger = logging.getLogger("nb_logger")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

file_handler = logging.FileHandler(log_path, encoding="utf-8")
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
