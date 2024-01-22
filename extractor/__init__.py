
import matplotlib as mpl
mpl.rc('image', cmap='magma')

Data_load_path = '../Distill.data.v2'
Saving_path = 'data'

from .pdf import pdf
doc = pdf()

from .test import study_line_fill_connection
from .test import study_disconnected_comp
from .test import study_buffering_by_paths, study_buffering_by_nodes
from .test import Clean_filling_strikes
from .test import clean_filled_strokes


from .tables_utils import clean_duplicates_paths

from .cleaning_grouping import Detect_unconnected_letters
from .cleaning_grouping import remove_borders
from .cleaning_grouping import find_boundingBoxes


from . import plotter

