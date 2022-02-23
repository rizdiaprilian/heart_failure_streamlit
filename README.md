# heart_failure_streamlit

Part of my personal project. Work in progress.

This project is referenced to a paper worked by Chicco and Jurman: 

•	Chicco, D., & Jurman, G. (2020). Machine learning can predict survival of patients with heart failure from serum creatinine and ejection fraction alone. BMC Medical Informatics and Decision Making, 20(1), 1-16.

More information is available in this link: https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records

## Structures
* `src/` - src folder that organize modules and class
* `EDA/` - has notebooks that gives outlines of heart failure dataset
* `main.py` - Streamlit-powered app that fit the learning and make prediction on data retrieved from postgre-SQL in local machine.
* `save_to_gsheets.py` - Small utility script that saves a bunch of metadata about an uploaded image to a Google Sheet (this will likely move into a dedicated `utils/` folder later on.
* `requirements.txt` - A text file with the dependency requirements for this project.
