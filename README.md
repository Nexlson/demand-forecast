# UK Power Usage Forecasting
This repository contains jupyter notebooks and streamlit demo app on UK power usage forecasting use case 

### Problem Statement
Power usage of a city can be influenced by weather and has a seasonal effect. Can we predict accurately the future power usage based on some weather data and power usage in the past?


## Getting Started
### Download Data
[Google Drive Link](https://drive.google.com/file/d/152qwa-oTBSXTXHZGJnxmJQ_BQLpQIccr/view?usp=sharing)
- download the csv file and put into data directory

### Environment Setup
#### Setup Conda Environments
```
conda env create -f environment.yml
```
### Activate Environment
```
conda activate power-usage-forecasting-labs
```
### Install Streamlit
```
pip install streamlit
```
#### Run Streamlit in Localhost
```
streamlit run demo.py
```


## Streamlit Demo
### Web Application
'''https://share.streamlit.io/xin133/demo_puf/main/demo.py''' - Use streamlit.io <br>
This system use lstm to predict power usage of a city. User can choose hyperparameter to tune the model.

## Acknowledgments and References
* [Dataset used](https://www.kaggle.com/jeanmidev/smart-meters-in-london)
* [Description on dataset](https://medium.com/@boitemailjeanmid/smart-meters-in-london-part1-description-and-first-insights-jean-michel-d-db97af2de71b)
