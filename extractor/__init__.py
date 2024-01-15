
import matplotlib as mpl
mpl.rc('image', cmap='magma')

Data_load_path = '../data'
Saving_path = 'data'

from .pdf import pdf
doc = pdf()

from .test import study_line_fill_connection
from .test import study_paths_svg
from .test import study_disconnected_comp
from .test import study_buffering_by_paths, study_buffering_by_nodes
from .test import plot_full_dwg
from .test import Clean_filling_strikes
from .test import Detect_unconnected_letters
from .test import clean_filled_strokes

from .tables_utils import clean_duplicates_paths

# from . import test as tester

from . import plotter

