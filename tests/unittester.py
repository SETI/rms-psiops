##########################################################################################
# tests/unittester.py
##########################################################################################

import unittest

from tests.test_ishift   import *
from tests.test_maximum  import *
from tests.test_mean     import *
from tests.test_median   import *
from tests.test_minimum  import *
from tests.test_resample import *
from tests.test_reshape  import *
from tests.test_rotate   import *
from tests.test_shift    import *
from tests.test_stdev    import *
from tests.test_utils    import *
from tests.test_zoom     import *

from tests.test_resize   import *   # removed from image_ops but retained inside tests/
                                    # for cross-testing with other functions.

############################################
# Execute from command line...
############################################

if __name__ == '__main__':
    unittest.main(verbosity=2)

##########################################################################################
