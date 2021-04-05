# UK Power Usage Forecasting
### Problem Statement
Power usage of a city can be influenced by weather and has a seasonal effect. Can we predict accurately the future power usage based on some weather data and power usage in the past?

### Environment Setup
#### Setup conda environments
```
conda env create -f environment.yml
```
### Activate environment
```
conda activate power-usage-forecasting-labs
```
### Install streamlit
```
pip install streamlit
```

### Download data
[Google Drive Link](https://drive.google.com/file/d/152qwa-oTBSXTXHZGJnxmJQ_BQLpQIccr/view?usp=sharing)
- download the csv file and put into data directory

### Web Application
### Initialize streamlit app 
```
streamlit run demo.py
```
'''https://streamlit.io/''' - Use streamlit.io 
This system use lstm to predict power usage of a city. User can choose hyperparameter to tune the model

### Acknowledgments and References
* [Dataset used](https://www.kaggle.com/jeanmidev/smart-meters-in-london)
* [Description on dataset](https://medium.com/@boitemailjeanmid/smart-meters-in-london-part1-description-and-first-insights-jean-michel-d-db97af2de71b)


