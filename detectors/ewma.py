"""
The Tornado Framework
By Ali Pesaranghader
University of Ottawa, Ontario, Canada
E-mail: apesaran -at- uottawa -dot- ca / alipsgh -at- gmail -dot- com
---
*** The Exponentially Weighted Moving Average (EWMA) Method Implementation ***
Paper: Ross, Gordon J., et al. "Exponentially weighted moving average charts for detecting concept drift."
Published in: Pattern Recognition Letters 33.2 (2012): 191-198.
URL: https://arxiv.org/pdf/1212.6018.pdf
"""

import math
from detectors.super_detector import SUPER_DETECTOR 

class EWMA(SUPER_DETECTOR):
    """The Exponentially Weighted Moving Average (EWMA) drift detection method class."""
    
    def __init__(self, min_instance=30, lambda_=0.2, c=1, w=0.5):
        '''
        The Exponentially Weighted Moving Average (EWMA) drift detection method class.
        :param: min_instance: quantity of instance to start detect a concept drift
        '''

        self.MINIMUM_NUM_INSTANCES = min_instance

        self.t = 1.0
        self.sum = 0.0
        self.sigma_xt = 0.0
        self.sigma_zt = 0.0
        self.z_t = 0.0
        self.lambda_ = lambda_
        self.L_t = c
        self.w = w

    def run(self, prediction):
        '''
        method to update the parameters of ewma
        :param: prediction: true if the prediction is correct, otherwise no
        '''

        prediction = 1 if prediction is False else 0

        warning_status = False
        drift_status = False

        # 1. UPDATING STATS
        self.sum += prediction
        self.sigma_xt = self.sum / self.t
        self.sigma_zt = math.sqrt(self.sigma_xt * (1.0 - self.sigma_xt) * self.lambda_ * (1.0 - math.pow(1.0 - self.lambda_, 2.0 * self.t)) / (2.0 - self.lambda_))
        self.t += 1

        self.z_t += self.lambda_ * (prediction - self.z_t)
        #L_t = 3.97 - 6.56 * self.sigma_xt + 48.73 * math.pow(self.sigma_xt, 3) - 330.13 * math.pow(self.sigma_xt, 5) + 848.18 * math.pow(self.sigma_xt, 7)

        # 2. UPDATING WARNING AND DRIFT STATUSES
        if self.t < self.MINIMUM_NUM_INSTANCES:
            return False, False

        if self.z_t > self.sigma_xt + self.L_t * self.sigma_zt:
            drift_status = True
        elif self.z_t > self.sigma_xt + self.w * self.L_t * self.sigma_zt:
            warning_status = True

        return warning_status, drift_status

    def reset(self):
        '''
        method to reset the detector
        '''
        
        self.t = 1
        self.sum = 0
        self.sigma_xt = 0
        self.sigma_zt = 0
        self.z_t = 0

