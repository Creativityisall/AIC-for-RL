from datetime import datetime
import logging
import os

# 设置日志级别
ll = 'INFO'

# 创建日志储存文件夹
log_dir = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

# 创建日志记录器
logger = logging.getLogger(__name__)
logger.propagate = False  # 防止日志输出重复
logger.setLevel(getattr(logging, ll))  # 设置日志级别

# 创建一个FileHandler，用于将日志写入文件
cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
file_handler = logging.FileHandler(os.path.join(log_dir, f'{cur_time}.log'), encoding="utf-8")  # 指定日志文件路径
file_handler.setLevel(getattr(logging, ll))  # 设置FileHandler的日志级别

# 创建一个StreamHandler，用于将日志输出到控制台
stream_handler = logging.StreamHandler()  # 默认输出到标准输出
stream_handler.setLevel(logging.INFO)  # 设置StreamHandler的日志级别

# 创建一个日志格式器
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
file_handler.setFormatter(formatter)  # 将格式器添加到FileHandler
stream_handler.setFormatter(formatter)  # 将格式器添加到StreamHandler

# 将FileHandler和StreamHandler添加到日志记录器
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

try:
    import coloredlogs
    coloredlogs.install(level=getattr(logging, ll), logger=logger)
except ImportError:
    pass

if __name__ == '__main__':
    # 测试日志
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")