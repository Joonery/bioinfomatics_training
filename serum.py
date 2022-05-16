import pandas as pd

n = pd.Series([10,11,12,13])
print(n)
# 0    10
# 1    11
# 2    12
# 3    13

# 다음과 같이 출력됨.
print(n.keys()) # RangeIndex(start=0, stop=4, step=1)
print(n.values) # [10 11 12 13] 