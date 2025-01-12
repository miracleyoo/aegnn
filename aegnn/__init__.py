from . import asyncronous
from . import utils
from . import datasets
from . import models
from aegnn.utils.io import setup_environment
try:
    from . import visualize
except ModuleNotFoundError:
    import logging
    logging.warning("AEGNN Module imported without visualization tools")

# Setup default values for environment variables, if they have not been defined already.
# Consequently, when another system is used, other than the default system, the env variable
# can simply be changed prior to importing the `aegnn` module.
setup_environment({
    "AEGNN_DATA_DIR": "/data/storage/",
    "AEGNN_LOG_DIR": "/data/scratch/"
})
