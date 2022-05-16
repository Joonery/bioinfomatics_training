# 2차원 table을 index, column로 불러옴.
# 데이터 분석.

import pandas as pd
import numpy as np

########################################### Series 객체 만들기 ########
# Series 객체 / 유사 dict
s = pd.Series([2,-1,3,5])
print(s)
# 0에는 2 대응
# 1에는 -1 대응
# 2에는 3 대응
# 3에는 5 대응.

########################################### Series 객체의 연산 ########
s1 = s + [1000,2000,3000,4000] # 각 원소에 넣어주기
print(s1)

s2 = s + 1000 # 모든 원소에 더해주고
print(s2)

s3 = s < 0 # TF도 판별해줌
print(s3)

########################################### Series 객체의 인덱스 ########
# 인덱스 넣기
si = pd.Series([68, 83, 112, 68], index=["alice", "bob", "charles", "darwin"])
print(si)
print(si["bob"])   # 동일한 결과
print(si[1])       # 동일한 결과

# 슬라이싱 가능
s2.iloc[1:3] # 인덱스 로케이션. 무조건 인덱스를 줘야 함.
s2.loc["bob"] # 이렇게 하면 그냥 location (행 이름)을 찾아서 해줌.


# 주의점
surprise = pd.Series([1000, 1001, 1002, 1003])
surprise_slice = surprise[2:] # 슬라이스를 하면, 인덱스가 2부터 시작한다. 2~3만 갖겠지
# surprise_slice[0] # > 이건 못불러옴!  원본 참조라 2~3만 갖고있으니까. (절대영역)
surprise_slice.iloc[0] # > 이렇게는 불러올 수 ㅣㅇㅆ음. 사실상 2~3이 각각 0~1이니까 0을 불러줌. (상대영역)



########################################### Series 객체와 dict ########

# dict와 의 변환 가능
weights = {"alice": 68, "bob": 83, "colin": 86, "darwin": 68} # dict를 만들어주고
sw1 = pd.Series(weights) # 전체 변환
sw2 = pd.Series(weights, index = ["colin", "alice"]) # 일부 변환
print(sw1)
print(sw2)


########################################### Series 객체의 정렬 ########


s1 = pd.Series([1000,2000,4000,6000], index=["오뎅", "떡볶이", "유부김밥", "라볶이"])
s2 = pd.Series([1000,2000,5000,6000], index=["오뎅", "떡볶이", "참치김밥", "라볶이"])
print(s1+s2) # 합집합이 됨. 공통으로 없는 건 NaN값이 되고.

########################################### Series 객체의 스칼라화 ########

meaning = pd.Series(42, ["life", "universe", "everything"]) # 리스트를 따로 넣어주지 않고 한번에 초기화
print(meaning)

########################################### Series 객체 이름 ########

s6 = pd.Series([83, 68], index=["bob", "alice"], name="weights") # 이름 지정 가능
s6

########################################### Series 객체 그래프로 만들기 ########

import matplotlib.pyplot as plt
temperatures = [4.4,5.1,6.1,6.2,6.1,6.1,5.7,5.2,4.7,4.1,3.9,3.5] # 일반적으로 이렇게 만들겠네. 리스트를 주고
s7 = pd.Series(temperatures, name="Temperature") # 이걸 시리즈로 만들어서
s7.plot() # 플롯을 그리고
plt.show() # 보여라~

s7.plot? # 이라고 치면 넘길 수 있는 파라미터를 보여줌.

########################################### 시간 다루기 (응 안써~) ########

dates = pd.date_range('2016/10/29 5:30pm', periods=12, freq='H') # 총 12개를 만든다
print(dates)

########################################### Dataframe (2차원) ########

# dict로 만들어서 전달하기 (이게 제일 괜찮은 방법인듯?)
people_dict = {
    "weight": pd.Series([68, 83, 112], index=["alice", "bob", "charles"]),
    "birthyear": pd.Series([1984, 1985, 1992], index=["bob", "alice", "charles"], name="year"),
    "children": pd.Series([0, 3], index=["charles", "bob"]),
    "hobby": pd.Series(["Biking", "Dancing"], index=["alice", "bob"]),
}
people = pd.DataFrame(people_dict)
print(people) # 표로 출력됨



print(people["birthyear"]) # 특정 column만 추출 가능
print(people[["birthyear", "hobby"]]) # 여러 열 추출 가능


# 특정 row/col만 추출해서 새로 만들기
d2 = pd.DataFrame(
        people_dict,
        columns=["birthyear", "weight", "height"],
        index=["bob", "alice", "eugene"] # 유진이라는 index는 없으므로 NaN으로 채워짐.
        )
print(d2)



# value list를 따로 넘기기
values = [  [1985, np.nan, "Biking",   68],
            [1984, 3,      "Dancing",  83],
            [1992, 0,      np.nan,    112]
         ]
d3 = pd.DataFrame(
        values,
        columns=["birthyear", "children", "hobby", "weight"],
        index=["alice", "bob", "charles"]
     )
print(d3)


# dict 안에 dict를 넣어도 됨. 복잡해서 이걸론 안할듯..
people = pd.DataFrame({
    "birthyear": {"alice":1985, "bob": 1984, "charles": 1992},
    "hobby": {"alice":"Biking", "bob": "Dancing"},
    "weight": {"alice":68, "bob": 83, "charles": 112},
    "children": {"bob": 3, "charles": 0}
})
people

# 전치 .T하면 된다~




# 멀티인덱싱 (굳이?)
d5 = pd.DataFrame(
  {
    ("public", "birthyear"):
        {("Paris","alice"):1985, ("Paris","bob"): 1984, ("London","charles"): 1992},
    ("public", "hobby"):
        {("Paris","alice"):"Biking", ("Paris","bob"): "Dancing"},
    ("private", "weight"):
        {("Paris","alice"):68, ("Paris","bob"): 83, ("London","charles"): 112},
    ("private", "children"):
        {("Paris", "alice"):np.nan, ("Paris","bob"): 3, ("London","charles"): 0}
  }
)
print(d5)

# Stack, unstack(). 굳이?




# 행 참조
people.loc["charles"] # 해당 row 불러오기
people.iloc[2] # 3번째 row
people.iloc[1:3] # ????????????? index out of range 오류가 안뜨나?
people[np.array([True, False, True])]





# col 추가와 삭제
people["age"] = 2018 - people["birthyear"]  # "age" 열을 추가합니다
people["over 30"] = people["age"] > 30      # "over 30" 열을 추가합니다
birthyears = people.pop("birthyear")        # birthyear라는 뽑아내고 삭제
del people["children"]

# 추가할 열을 직접 series로 지정
people["pets"] = pd.Series({"bob": 0, "charles": 5, "eugene":1})  # alice 누락됨, eugene은 무시됨

# 일반적으로는 맨 끝에 추가되나 추가될 곳을 지정해줄수도 있음.
people.insert(1, "height", [172, 181, 185])

# assign() 메서드를 쓸수도 있는데 잘 안쓸거같으니 필요하면 참조하세여~






########################################### Dataframe의 저장과 로딩 ########





########################################### Dataframe과 SQL ########





