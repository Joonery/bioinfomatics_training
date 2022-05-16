import numpy as np

a = np.random.rand(2,3)


# .npy로 저장
# np.save("my_array", a) # .npy로 저장됨.



# 로딩
# a_loaded = np.load("my_array.npy")
# print(a_loaded)


# .csv로 저장
np.savetxt("my_array.csv", a)

# 열기
with open("my_array.csv", "rt") as f:
    print(f.read())
