# BoardGameRecommender

This is source code required for our Bachelor's thesis at University of Wrocław.
Goal was to research already existing solutions for creating recommending systems on the example of board games
and propose improvements.

## Table of contents
- [Setup](#setup)
- [Project structure](#project-structure)
- [Authors](#authors)


## Setup
1. Create virtual environment:

    `python -m venv venv && source venv/bin/activate`

2. Install required dependencies:

    `pip install -r requirements.txt`

3. Install our modified version of LightFM:

    `cd lightfm-custom && python setup.py cythonize && pip install -e .`

4. To acquire raw data contact either us or Markus Shepherd from BoardGameGeek.
    Then to preprocess it, run the pipeline manually in order described in Appendix of our thesis or use DVC:

    `dvc repro`

5. Finally, run following command to properly set up PYTHONPATH environmental variable:

    `export PYTHONPATH=$PYTHONPATH:.`

## Project structure
The project consists of following directories:
- ``data`` - place for storing all the data
- ``data-pipeline`` - includes scripts making up the data processing pipeline
- ``lightfm-custom`` - includes modified version of LightFM scikit; needs to be installed as described in [Setup](#setup)
- ``notebooks`` - includes all the Jupyter notebooks we used to produce our results
- ``recommender`` - includes utilities for calculating our metric and evaluating the models

## Authors
Adrian Urbański, Maria Wyrzykowska
