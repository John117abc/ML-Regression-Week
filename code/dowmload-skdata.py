import os
import tarfile
import pandas as pd
from sklearn.utils import Bunch
from sklearn.datasets import get_data_home
import joblib

data_home = get_data_home()
cache_dir = os.path.join(data_home, "california_housing")
tgz_path = os.path.join(cache_dir, "california_housing.tgz")
print(cache_dir)
# 解压
with tarfile.open(tgz_path, "r:gz") as tar:
    tar.extractall(path=cache_dir)

csv_path = os.path.join(cache_dir, "california_housing.csv")
df = pd.read_csv(csv_path)

# 构造 Bunch
bunch = Bunch(
    data=df.drop(columns=["MedHouseVal"]).values,
    target=df["MedHouseVal"].values,
    feature_names=list(df.columns[:-1]),
    DESCR="California housing dataset."
)

# 保存为 sklearn 认可的 .pkz 缓存
joblib.dump(bunch, os.path.join(cache_dir, "california_housing.pkz"))
print("✅ 缓存文件已生成，现在可以离线使用 fetch_california_housing()！")