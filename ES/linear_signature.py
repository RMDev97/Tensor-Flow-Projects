"""
This model will implement a basic linear classifier on the MNIST dataset using the ES model
1. We treat the inputs of the dataset as being a discrete path through the a 2 dimensional tensor space
2. We apply the lead-lag transformation in order to account for quadratic variation of the path and to
transform it into a continuous path
3. We calculate the signature of the path (truncated to some finite order N)
4. using the truncated Nth order signature of the lead-lag transformed path as a feature set,
we train a softmax classifier (trained with the ADAM optimisation algorithm)
5. we will then extract the details of this into a class which can be used elsewhere in code
"""

import tensorflow as tf
import numpy as np

