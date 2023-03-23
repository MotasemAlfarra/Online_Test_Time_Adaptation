from tta_methods.basic import Basic_Wrapper
from tta_methods.tent import Tent
from tta_methods.eata import EATA
from tta_methods.cotta import CoTTA
from tta_methods.memo import Memo
from tta_methods.ttac import TTAC
from tta_methods.adabn import AdaBn
from tta_methods.shot_im import SHOTIM
from tta_methods.shot import SHOT
from tta_methods.lame import LAME
from tta_methods.bn_adaptation import BN_Adaptation
from tta_methods.pl import PSEUDOLABEL
from tta_methods.sar import SAR
from tta_methods.dda import DDA


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