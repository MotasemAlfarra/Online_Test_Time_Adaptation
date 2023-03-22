from tte_methods.basic import Basic_Wrapper
from tte_methods.tent import Tent
from tte_methods.eata import EATA
from tte_methods.cotta import CoTTA
from tte_methods.memo import Memo
from tte_methods.ttac import TTAC
from tte_methods.adabn import AdaBn
from tte_methods.shot_im import SHOTIM
from tte_methods.shot import SHOT
from tte_methods.lame import LAME
from tte_methods.bn_adaptation import BN_Adaptation
from tte_methods.pl import PSEUDOLABEL
from tte_methods.sar import SAR
from tte_methods.dda import DDA


_all_methods = {
    'basic': Basic_Wrapper,
    'tent': Tent,
    'eta': EATA,
    'eata': EATA,
    'cotta':CoTTA,
    'memo': Memo,
    'ttac_nq': TTAC,
    'adabn': AdaBn,
    'shotim': SHOTIM,
    'shot': SHOT,
    'lame': LAME,
    'bn_adaptation': BN_Adaptation,
    'pl': PSEUDOLABEL,
    'sar': SAR,
    'dda': DDA,
}