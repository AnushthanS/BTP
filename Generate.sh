#! /bin/bash
# This script will generate the python venv named py with all the dependencies
python3 -m venv py
source py/bin/activate
pip install -r requirements.txt
exit