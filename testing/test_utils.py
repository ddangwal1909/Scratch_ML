_TEST_DATA_PATH = f"/testing/test_data"

from common_utils import *

def get_test_data(file_name:str,target_col_name="target"):
    df = pd.read_csv(f"{_TEST_DATA_PATH}/{file_name}.csv")
    train_cols = [_ for _ in df.columns if _!=target_col_name]
    x = df[train_cols].to_numpy()
    y = df[target_col_name].to_numpy()

    return x,y



