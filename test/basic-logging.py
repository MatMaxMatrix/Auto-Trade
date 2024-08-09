#Here we try to explain the logging module in python
#%%
import logging

logging.basicConfig(level=logging.INFO, filename="log.log", filemode="w",
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug("debug")
logging.info("info")
logging.warning("warning")
logging.error("error")
logging.critical("critical")

# %%
x = 2
logging.info(f"x is {x}")


try:
    1/0
except ZeroDivisionError as e:
    logging.error(f"Error", exc_info=True)


#
#another way to do the above log

try:
    1/0
except ZeroDivisionError as e:
    logging.exception("second-example-ERROR")


#next step is to handler to save the log to a differnt file from the default file
#%%
logger = logging.getLogger(__name__)
handler = logging.FileHandler("test.log")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.info("test")