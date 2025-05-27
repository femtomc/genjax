import time

from core import GibbsSampler
from dataloading import get_blinker_4x4

blinker_small = get_blinker_4x4()

print(GibbsSampler(blinker_small, 0.3).get_initial_state())
