"""Climate data pre-processing"""
import warnings
warnings.warn('The module `oggm.core.gcm_climate` has moved to '
              'oggm.shop.gcm_climate. This compatibility module will be '
              'removed in future OGGM versions', FutureWarning)
from oggm.shop.gcm_climate import (process_gcm_data, process_cmip5_data,
                                   process_cesm_data, process_cmip_data)
