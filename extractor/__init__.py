
import matplotlib as mpl
mpl.rc('image', cmap='magma')

# PDF_Data_path = '../Distill.data.v2'
Data_logic_load_path = '../Distill.data.v2/LOGIC'
Data_pid_load_path = '../Distill.data.v2/PID'

Data_load_path = Data_logic_load_path

Saving_path = 'data/LOGIC'
LS_path = 'LS'

from .pdf import pdf
doc = pdf()

from .test import study_line_fill_connection
from .test import study_disconnected_comp
from .test import study_buffering_by_paths, study_buffering_by_nodes
from .test import Clean_filling_strikes
from .test import clean_filled_strokes
from .test import study_clustering
from .test import extrct_features
from .test import group_clustering


from .tables_utils import clean_duplicates_paths

from .cleaning_grouping import Detect_unconnected_letters
from .cleaning_grouping import remove_borders
from .cleaning_grouping import find_boundingBoxes

from .logic_modules.detection_methods import detect_logic_components

from .PID_functions import detect_rectangles
from .PID_functions import detect_rectangles2


from . import plotter

