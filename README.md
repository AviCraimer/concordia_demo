## Using vevn:

~/shared_python_venvs/ai

To Activate:
`source ~/shared_python_venvs/ai/bin/activate`

## To Install Packages:

Install external packages
`pip install -r requirements.txt`

`pip install -r dev-requirements.txt`

Install my own package
`pip install -e .`

Single Command
`pip install -r dev-requirements.txt && pip install -r requirements.txt && pip install -e .`

## To Run Jupyter Notebook

jupyter notebook

## Type Hints

To make some type annoations work properly you need to add:
`from __future__ import annotations`
At the top of the file.
