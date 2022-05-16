import numpy as np
from numpy.core.shape_base import stack


# 행렬을 만드는법 ==============================================
# zeros, ones
a = np.zeros(5) # 1차원 개수만 전달 가능.
a = np.zeros((3,4)) # 튜플로 행렬 전달 가능

a = np.ones(5) # 1차원 개수만 전달 가능.
a = np.ones((3,4)) # 튜플로 행렬 전달 가능


# 일반적으로 쓰이는 느낌
a = np.arange(24).reshape(2,3,4) # 0~23의 숫자를 2*3*4 matrix에 배열한다.



# 행렬의 메타데이터를 뽑아내는 법. 
a.shape # 행렬을 튜플로 뽑아줌.
a.ndim # dimension의 개수. 2차원 3차원
a.size # 크기 (3*4행렬이면 12)
type(a) # 

# ============================================
# rand

# np.full / np.empty 

# arange

# reshape

# 산술연산

# 브로드캐스팅 잘 익혀두기


#

# 행렬 안에 있는 평균, min, max, sum, std, var 등 구하기. 메소드로 가능

    # a.min
    # a.max
    # a.sum
    # a.std
    # a.var

    # 기본적으로 약간 np.arange(24).reshape(2,3,4) 이런식으로 만드는듯
    # 0~23까지를 만들고, 2*3*4 로 모양 재배열함.

# ============================================
# 조건연산자

m = np.array([20, -5, 30, 40])
m < [15, 16, 35, 36]

또는 
m < 25 로 써도 [T,T, F, F]로 반환됨.



# 이항일반함수
행렬 두개에 적용 가능한 애들.
np.greater(a,b)
np.maximum(a,b)


# 1차원 인덱싱
# 리스트와 똑같이 되며, 원소 수정도 가능. (start, stop, step)
a = np.array([1,4,5,6,7,8,1,4])
a[3]

다만 arary의 형태를 맞춰주지 않으면 오류 발생.
a[2:5] = -1 # 2~5까지 -1이 됨.
a[15] 없는 메모리를 참조할 순 없음.



### 인덱싱 시 주의할 것 ### 
주의할 것! 만일 슬라이싱해서 복사한 애들은 진짜 복사한 게 아니라 원본 참조하고있는거라서,
a_slice = a[2:6]  # a_slice의 값을 바꾸면 원본의 값도 바뀜. a도 바뀜;;


진짜로 복사를 하려면 .copy를 써야 함.
real_copy = a[2:6].copy



# n차원 인덱싱
b = np.arange(24).reshape(3*8)
b[1,2] # 1행 2열
b[1, :] # 1헹 모든열
b[:, 1] # 모든행 1열




# Fancy 인덱싱

# 고차원

# 생략부호 ...

# 불리언 인덱싱



# 반복 =-===========================================
c = np.arange(24).reshape(2, 3, 4)

# 행렬 단위로 반복할 때
for m in c:
    print("아이템:")
    print(m)


# flat~ 하게 반복할 때. 그냥 리스트처럼.
for i in c.flat:
    print("아이템:", i)



# 배열 쌓기 ============================================
일단 만들어보자.
q1 = np.full((3,4), 1.0)
q2 = np.full((4,4), 2.0)
q3 = np.full((3,4), 3.0)

vstack은 수직으로 쌓음. (열의 개수가 같아야 함)
q4 = np.vstack((q1, q2, q3))

hstack은 수평으로 쌓음. (행의 개수가 같아야 함)
q5 = np.hstack((q1, q3))

행/열 개수가 쌓을 수 없을 경우 dimension오류가 남.

q7 = np.concatenate((q1, q2, q3), axis=0) # 지정한 축으로 쌓도록 함.

q8 = np.stack((q1, q3)) # 새로운 축을 만들어서 쌓음. 2차원 도형을 3차원 축으로 쌓는 것처럼. 직사각형을 정육면체로.
# 이걸 하려면 정확히 동일한 행/열 크기어야 함.


# 배열 분할 ============================================
r = np.arange(24).reshape(6,4)

r1, r2, r3 = np.vsplit(r, 3) # 3개로 분할
r4, r5 = np.hsplit(r, 2) # 2개로 분할


# 배열 전치 transpose ============================================
t = np.arange(24).reshape(4,2,3)

# 디폴트는 그냥 뒤집기
t2 = t.transpose()

# 만일 차원의 뒤집을 순서를 지정하고 싶은 경우
t1 = t.transpose((1,2,0)) 

# 그냥 두 축만 바꿔줌. 
t3 = t.swapaxes(0,1)

또는, m1.T로 쓰면 전치행렬이 됨.


다른 변수에 transpose() 메소드를 써서 전치행렬을 넣어도 슬라이싱처럼 원본을 참조하나요? 네. 바꾸려면 copy 쓰셈
m3 = m1.T.copy()

# 행렬곱
n1 = np.arange(10).reshape(2,5) # 곱해지는 것의 열 5 
n2 = np.arange(15).reshape(5,3) # 곱할 것의 행 5 이 같아야 한다.
n1.dot(n2)

n1*n2는 원소별 곱셈이고, 애초에 크기도 안맞음


# 역행렬




# 배열 전치 transpose ============================================

a = np.random.rand(2,3)

np.save("my_array", a) # 작업 디렉토리에 .npy로 저장됨.
np.save("Desktop/my_array", a) # 경로 지정 가능.


# 구드를 마운트해서 넣을수도 있음(코랩 only)
from google.colab import drive
drive.mount('/content/drive')
np.save("/content/drive/my_array", a) # 작업 디렉토리에 .npy로 저장됨.


# 읽으려고 열 수 있고
with open("my_array.npy", "rb") as f:
    content = f.read()
print(content)


# 로딩 가능
a_loaded = np.load("my_array.npy")
print(a_loaded)


# 텍스트 파일로 내보내기 (comma separated vector)
np.savetxt("my_array.csv", a)
np.savetxt("my_array.csv", a, delimiter=",") # 구분자를 콤마로 불러옴.

with open("my_array.csv", "rt") as f:
    print(f.read())


a_loaded = np.loadtxt("my_array.csv", delimiter=",") # 구분자를 콤마로 불ㄹ옴.
print(a_loaded)


np.savez("my_arrays")