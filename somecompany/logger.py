"""Simple logger setup"""

import logging
import os

LOGLEVEL = os.getenv("LOGLEVEL", "WARNING").upper()
logging.basicConfig(
    level=LOGLEVEL,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
)
# suppress these libraries anyways, they are often very verbose even at INFO level
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)
