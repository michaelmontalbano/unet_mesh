"""By using execfile(this_file, dict(__file__=this_file)) you will
activate this virtualenv environment.
This can be used when you must use an existing Python interpreter, not
the virtualenv bin/python
"""
import os

activate_this = "/home/michaelm/.virtualenvs/new/bin/activate_this.py"

with open(activate_this) as f:
    code = compile(f.read(), activate_this, 'exec')
    exec(code, dict(__file__=activate_this))

import pandas as pd

def read_pickle_file(file):
    pickle_data = pd.read_pickle(file)
    return pickle_data['true_testing']