@Author: Kai Ching Suen
@Data: 4/20/2019

START:

# In project root
$ pip freeze > requirements.txt
# And to install the packages
$ pip install -r requirements.txt

TEAM:
Kai Ching Suen
Kaito Kumagia
Manan Duggle
Unais Ibarahim

PROJECT GOALS:
this project is to test a data source (signal, second column in data.csv) \
  which claims to be predictive of future returns of the SP500 index \
  (spy_close_price, third column in data.csv).

STEPS:
  1. data cleaning (1 person need)
  2. analysis (2 to perform all analysis, and compare results)
  3. summary (1 person needed)

INSTRUCTIONS:
  1. DATA CLEANING
    NOTE: Assume all values in data.csv are potentially suspect.
    a) Please identify any errors in the data
    b) Flag them with a note
    c) And suggest a corrected value or if advisable \
        (may choose to ignore them for purposes of your analysis)
    d) Explain what types of analysis you did to identify the errors
    e) Provide any assumptions/intuition/formulas/scripts you may have used \
        to help you find them

  2. ANALYSIS
    NOTE: Analysis could take qualitative, to linear regression to recurrent \
    neural networks, and everything in between.
    a)  Use the above technique(s) to perform an analysis of the predictive \
        power of signal with respect to spy_close_price (third column in data.csv) \
        under 3 considerations:
            -  your general familiarity
            -  their potential for success on this task
            - the time involved
    b) Please share all your ideas/attempts, even if they proved less than successful
    c) Guess(es) as to why they didn't work or how to improve them would be \
      great as well

  3. SUMMARY
    1. Document the experiment(s) you performed (including relevant code, \
        package references, etc)
    2. Summarize your conclusions about the viability and shortcomings of this \
        signal as a predictor of spy_close_price, including any materials you \
        feel are appropriate to support your conclusions (eg, graphs, tables, etc)
    3. Use jupyter notebook. If there were other experiments you didn't \
        have time to perform, or future avenues of work you might like to \
        pursue, please discuss those as well (we may work on these ideas \
        together as a follow-up).

FILES:

├── README.md          <- Front page of the project. Let everyone
│                         know the major points.
│
├── models             <- Trained and serialized models, model
│                         predictions, or model summaries.
│
├── notebooks          <- Jupyter notebooks. Use set naming
│                         E.g. `1.2-rd-data-exploration`.
│
├── reports            <- HTML, PDF, and LaTeX.
│   └── figures        <- Generated figures.
│   └── docs
│
├── requirements.txt   <- File for reproducing the environment
│                         `$ pip freeze > requirements.txt`
├── data
│   ├── processed      <- The final data sets for modelling.
│   └── raw            <- The original, immutable data.
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module.
    │
    ├── utility <- General functions to import.
    |    └── custom_func.py
    │
    ├── features       <- Scripts raw data into features for
    │   │                 modeling.
    │   └── feature_builder.py
    │
    ├── models         <- Scripts to train models and then use
    │   │                 trained models to make predictions.
    │   │
    │   ├── prediction_controller.py
    │   └── train_controller.py
    │
    └── visualizations            <- Scripts to create visualizations.
        └── vizualizer.py

REFERENCE:
1. project file structures:
  https://medium.com/@rrfd/cookiecutter-data-science-organize-your-projects-atom-and-jupyter-2be7862f487e
