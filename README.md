# Streamlit Assignment
Created by Muhammad Farzam Khan.

## Files
- `Employee_Churn_MLOps_Assignment.ipynb`: Complete executed Jupyter notebook with schema export, Random Forest pipeline, evaluation, and saved-pipeline inference.
- `train_pipeline.py`: Equivalent standalone training script.
- `Employee(2).csv`: The supplied 4,653-row employee churn dataset.
- `my_feature_dict.pkl`: Exported categorical options and numerical metadata.
- `pipeline.pkl`: Fitted Random Forest pipeline.
- `app.py`: Streamlit app loading both pickle files.
- `requirements.txt`: Required libraries. Use a matching scikit-learn version for the pretrained pickle.

## Run locally
1. Open a terminal in this folder.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Optional retraining: `python train_pipeline.py` or run the notebook cells top-to-bottom.
4. Start the web app: `streamlit run app.py`
5. Enter categorical and numerical values and click **Predict Employee Churn**.

## Dataset / modeling choices
Target `LeaveOrNot`: `0` = stay; `1` = leave. The 8 predictors are split into 5 categorical (`Education`, `City`, `Gender`, `EverBenched`, `PaymentTier`) and 3 numerical (`Age`, `ExperienceInCurrentDomain`, `JoiningYear`). As in the class notebook, the pipeline uses ColumnTransformer, OneHotEncoder, StandardScaler, and RandomForestClassifier. Missing numeric values, if any, would use median imputation.


