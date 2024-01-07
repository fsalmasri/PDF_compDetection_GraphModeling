
import matplotlib as mpl
mpl.rc('image', cmap='magma')

Data_load_path = '../data'
Saving_path = 'data'

from .pdf import pdf
doc = pdf()

from .test import study_pathes
from .test import study_paths_svg

from . import plotter

