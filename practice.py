lll = ["몽쉘", "사과", "몽쉘", "사과", '믹스', "캔디"]
g = ['과자', '오예스']
print(lll)
lll[1] = "사과쥬스"
print(lll[0:5])

lll.remove("사과")
print(lll)

lll.append("젤리")
print(lll)



lll.extend(g)
print(lll)



a = ["사과"]
b = ["Candy"]

a.extend(b)
print(a)


a.append("참외")
print(a)

my_tuple = ("삼", "사", "오")
# 값들을 묶는것을 패킹 이라고한다.
(t, y, i) = my_tuple # 언패킹
print(my_tuple)

t = my_tuple[0]
y = my_tuple[1]
i = my_tuple[2]
print(my_tuple)

for i in range(0, 11):
    print(i)

    # numbers = i[]
# print(numbers)
# (1, 2, *3) = numbers
# print(numbers)


################################ 세트 ######################################
# 세트는 순서를 보장되지 않는다. 그러므로 index가 적용이 되지 않는다
A = {"돈가스", "보쌈", "제육덮밥"}
B_set = {"짬뽕", "초밥", "제육덮밥"}
# 두 사람이 공통으로 좋아하는 음식( = 교집합)
print(A.intersection(B_set)) # A와 B 공통
# A 만 좋아하는 음식( = 차집합 )
print(A.difference(B_set)) # A만 추리기

A.add("닭갈비") # 특정 값 추가
print(A)

A.remove("돈가스") # remove 특정 값 삭제
print(A)

A.clear() # 모든값 삭제
print(A)

B_set = {"짬뽕", "초밥", "제육덮밥"}
# del B_set # 완전 삭제 NameError: B_set 없다는 에러가 출력된다
# 리스트와 튜블 동일하다
print(B_set)

################################ 딕셔너리 ######################################
# dictionary 사전이란 의미
# key, value 항상 쌍으로 사용 key는 중복이 불가하다
# 딕셔너리 = {key1:vale1, key2:vale2, key3:vale3}\\
person1 = {
    '이름':'김기영',
    '나이':26,
    '키': 167,
    '몸무게':76
}

person2 = {
    '이름':'김영스',
    '나이':28,
    '키': 167,
    '몸무게':76
}
person3 = {
    '이름':'김가영',
    '나이':27,
    '키': 167,
    '몸무게':76
}

print(type(person3))
print(person1)
print(person2)
print(person2["나이"])
print(person3)
print(person3["몸무게"])
print(person3.get("별명")) # 없는 key에 접근하면 None 출력

person1["최종학력"] = '유치원' # 새로운 데이터 추가 새로운 값 추가
person1["키"]= 130 # 특정 key의 value를 수정 값 수정

# 여러 key 들의 value를 바꾸려면 person1.Update({'키':150, '몸무게':67})
person1.update({'키':150, '몸무게':67})
print(person1)

# key:value 삭제 ex) person1.pop('최종학력')
person1.pop('최종학력')
print(person1)

# 모든 데이터 삭제 ex) poerson1.clear()

# key 어떠한 key들이 있는지에 대한 정보 확인
print(person1.keys())

# value 어떠한 value들이 있는지에 대한 정보 확인
print(person1.values())

# value:value 들이 있는지에 대한 정보 확인
print(person1.items())

############################# 자료형 비교 #############################
"""
리스트  : 여러 값들을 순서대로 관리할때
튜플    : 값이 바뀔 일이 없거나, 바뀌면 안될때
세트    : 값의 존재 여부 중요할때, 중복은 안될때
딕셔너리 : key 를 통해 효율적으로 데이터를 관리하고 싶다.

             리스트     튜플        세트           딕셔너리
 선언      : lst=[]     t = ()      s = {}         s = {key:value}
 순서      : O          O           X              O (v3.7)
 중복허용  : O          O           X              X (key)
 접근      : lst[idx]   t[idx]      X              d[key]
                                                   d.get(key)
 수정      : O          X           X              O(value)

 추가      : append()   X           add()          d[key] = val
             insert()               update()       update()
             extend()

 삭제      : remover()  X           remove()       pop()
             pop()                  discard()      popitem
             clear()                pop()          clear()
                                    clear() 
튜플도 수정하는 방법이 있다
ex)
my_tuple = ('오예스', '몽셀')
my_list = list(my_tuple)
my_list.append('초코파이')
my_list = tuple(my_list)

튜플 <=> 리스트
tuple()  list()

리스트 중복값 허용
때로 중복값 제거 


"""
# 리스트
my_list = ['오예스', '몽셀', '초코파이', '초코파이', '초코파이']
print(my_list)
my_list = set(my_list)
print(my_list)
my_list = list(my_list)
# 세트는 중복 X 순서 X
print(my_list)

# 또 다른 방법으로는 딕셔너리
# 딕셔너리 중복 X 순서 O
# 세트
# 순서가 중요할 경우 세트로 변환하면 안됨
# 리스트
my_list = ['오예스', '몽셀', '초코파이', '초코파이', '초코파이']
my_dic=dict.fromkeys(my_list)
print(my_dic)

my_list = list(my_dic)
print(my_list)

############################## if 조건문 #################################
# if = 만약에 ~ 라면
# else = 그렇지 않다면
today = '일요일'
if today == '일요일':
    print('게임 한판')
print('공부시작')

today = '화요일'
if today == '일요일':
    print('게임 한판')
print('공부시작')

# if:
#     이문장
# else:
#     저문장
# 다음문장

today = '일요일'
if today == '일요일':
    print('게임 한판')
else:
    print('폰 5분만')
print('공부시작')

today = '화요일'
if today == '일요일':
    print('게임 한판')
else:
    print('폰 5분만')
print('공부시작')
# if = 만약에 ~ 라면
# else = 그렇지 않다면

total = 5 # 총 인원
if total < 4:
    print('추가비용이 없음')
else:
    print('1인당 만원')
print("감사합니다")

# 복습하기
"""
 if = 만약에 ~ 라면
 else = 그렇지 않다면
 elif = 아니라고? 그럼 이건 어때?
 다른 조건을 다시 한번 확인하기 위한 용도로 사용합니다.
 앞에 조건이 참이 아닌 경우

# 비교연산자

> 왼쪽 값이 크다
< 왼쪽 값이 작다
>=     왼쪽값이 크거나 같다
<=     왼쪽값이 작거나 같다
==     값이 같다
!=     값이 같지 않다

is     값과 타입이 같다
is not 값과 타입이 같지않다
"""

# totoal1 = int(input())

# if totoal1 >= 9:
#     print('인원이 많습니다')
# elif totoal1 < 4:
#     print("인원이 너무 적습니다") 
# else:
#     print("인원이 적당합니다.")

# 첫번쩨 조건이 만족하지 않을 때 두번째 조건문 실행 두번째 조건문이 만족하지 않을때
# 마지막 조건문 실행
""" 
ex)
temp = 40 # 체온
if temp >= 39:
    print('고열입니다')
elif temp >= 38:
    prin("미열입니다")
else:
    print('정상입니다')

# 비교연산자

> 왼쪽 값이 크다
< 왼쪽 값이 작다
>=     왼쪽값이 크거나 같다
<=     왼쪽값이 작거나 같다
==     값이 같다
!=     값이 같지 않다

is     값과 타입이 같다
is not 값과 타입이 같지않다

"""
temp = 35 # 체온
if temp >= 39:
    print('고열입니다')
elif temp >= 38:
    print("미열입니다")
else:
    print('정상입니다')
"""
 if 중첩문
 if 조건1:
   이문장
    # if 조건2:
        저문장
다음문장
"""
yellow_card = 0
foul =  False

if foul:
    yellow_card += 1
    if yellow_card==2:
        print('경고누적 퇴장')
    else:
        print('휴...조심해야지')
else:
    print('주의')

min = 35 # 게임시간

if min > 20:
    print('게임많이 했네')
    if min > 40:
        print('당장 안꺼')
else:
    print('뭐해?')

######################### 반복문 for ###########################
"""
for x(일반변수) in 반복 범위 또는 대상:
    반복문 수행 문장

for x in range(10):
    print("팔 벌려 뛰기 해")

range(10)은 어떤 범위 내에 숫자들을 만들어 주는 기능

범위 설정
range(1, 10)
시작 값과 끝나는 값을 설정해준다
예를 들어 반복 회수만 입력시 10
0~9 까지의 수를 반환하지만
시작값인 start 에서 1을 넣어주면 1~10까지의 값을 반환하고 반복한다

1이상, 10미만
for x in range(start 1, stop 10):
    print(x)


범위 range(시작, 끝, 증가)

for x in range(start 1, stop 10, step 2):
    print(x)

"""
for x in range(10):
    print(x,f"팔 벌려 뛰기 해 {x}회")

for x in range(1, 10):
    print(x)

for x in range(1, 10, 2):
    print(x)
# (1,3,5,7,9)
# 1이상 10미만 2만큼 증가

for x in range(2, 10, 2):
    print(x)
# (2,4,6,8)
# 2이상 10미만 2만큼 증가

for x in range(3, 10, 3):
    print(x)
# (3,6,9)
# 3이상 10미만 3만큼 증가

for n in range(1, 31, 10):
    print(f"{n}번은 {n}번부터 {n+9}번 까지 모아줘")

################ 반복대상 #############
"""
for x(일반변수) in  반복대상:
    반복문 수행 문장
반복대상에는 리스트, 튜플, 딕셔너리


"""
# 리스트
my_list = [1,2,3]
for x in my_list:
    print(x)

# 튜플
my_tuple = (1,2,3)
for x in my_tuple:
    print(x)

# 딕셔너리

person = {'이름':'나귀염', '나이':7, '키':120, '몸무게':23}
# value 값으로 출력
for v in person.values():
    print(v)

# 키값으로 출력
for k in person.keys():
    print(k)

# items로 keys 와 value 함께 출력
for k,v in person.items():
    print(k,v)

################### 문자열 반복문 #################
fruit = 'apple'
for p in fruit:
    print(p)
# 문자열 내 하나씩 출력

################### while문 #################
"""
- ~ 하는 동안
- for 는 정해진 범위 또는 데이터를 순회하면서 반복
- while은 조건이 참인 동안 계속해서 반복하는 것

while 조건:
    반복 수행문


"""
max = 25 # 최대 허용 무게
weight = 0 # 현재 캐리어 무게
item = 3 # 각 짐의 무게

while weight + item <= max: # 캐리어에 짐을 더 넣어도 되는지 확인
    weight+= item
    print('짐을 추가합니다.')
print(f"총무게는 {weight} 입니다")

humen = 3
king = 0
maxm = 5

while king + maxm <= humen:
     king += maxm
     print('백성이 더 추가합니다.')
print(f"총 인구는 {king} 입니다")

max = 25 # 최대 허용 무게
weight = 0 # 현재 캐리어 무게
item = 3 # 각 짐의 무게

him =100
kim = 0
barbel = 5

while kim + barbel <= him:
    kim+=barbel
    print("무게를 추가합니다")
print(f"더 들수 있는 무게는 {kim} 개입니다")

runnuning = 180
jae = 0
run = 3

while jae + run <= runnuning:
    jae += run
    print("뛸 수 있는 시간이 점점 늘어납니다.")
print(f"최대로 뛸수 있는 시간은 {jae}분 입니다.")

# break 반복문 탈출 break문은 보통 if 조건문과 함께사용

drama = ["시즌1","시즌2","시즌3","시즌4","시즌5"]
for x in drama:
    if x == "시즌3":
        print('재미없대, 그만보자')
        break
    print(f"{x} 시청")


movis = ["나홀로집에", "레이오브썬", "나혼자상속자", "그래서우리는", "별이 빛나는ㅁ밤에"]
for i in movis:
    if i == "나혼자상속자":
        print("이 영화 별로라고해")
        break
    print(f"{i} 시청")


print("============================")
movis = ["나홀로집에", "레이오브썬", "나혼자상속자", "그래서우리는", "별이 빛나는밤에"]
for i in movis:
    if i == "나혼자상속자":
        print("재밌없대, 건너뛰자")
        continue
    print(f"{i} 시청")

# 10 % == 1


for x in range(10):
    if x % 2 == 1:
        continue
    print(x)
print("============================")

for i in range(10):
    if i % 2 == 1:
        continue
    print(i)

"""
if 조건문:
for 반복문:
while 반복문:
def 함수:
try 예외:
class 클래스
"""

# # 리콜 대상 제품 조회
# products = ['JOA-2020', 'JOA-2021', 'SIRO-2021', 'SIRO-2022',]
# recall = [] # 리콜 대상 제품 리스트
# for p in products:
#     if p.startswith('SIRO'): # 제품명이 SIRO 로 시작하는가?
#         recall.append(p) # 추가

# print(recall)


""" # List Comprehension
recall = [] # 리콜 대상 제품 리스트
for p in products:
    if p.startswith('SIRO'): # 제품명이 SIRO 로 시작하는가?
        recall.append(p) # 추가

- 이만큼 해당하는 부분을 딱 한줄로 해결하는 방법이 있다.

# List Comprehension
리스트 내에서 어떤 조건에 해당하는 데이터만 뽑아내거나
아니면 값을 바꿔서 새로운 리스트를 만들 떄 사용 할 수 있습니다.

일반적인 사용 문법은 List Comprehension
new_list = [변수활용 for 변수 in 반복대상 if 조건]


my_list[1,2,3,4,5] 값이 들어있다.
- List Comprehension 이용해서 3보다 큰 숫자를 뽑아서 새로운 리스트를 만들려고한다.
- 그러면 my_list를 순회하면서 값이 3보다 큰지를 확인해야된다.
- 그렇게 하기위해서는 for 반복문에서 했던것처럼 for x in 대상
- 즉 my_list를 먼저 적는다.
- 리스트 내의 값들이 순서대로 x에 들어 가는데
-  이 x를 임의로 바꾸는 작업을 할 수있다.
- 이때 for 앞에 x를 이용해 식을 적어주면 되는데

- 그냥 x라고 적으면 리스트와 값을 변경 없이 그대로 쓰겠다는게 되지만
- x+1 을하게되면 각 리스트에 1을 더한 값들
- x*3 을하게되면 각 리스트에 3을 곱한 값들

- x를 문자열로 바꾼 다음에 "번째"라는 글자를 붙이면
- str(x) + '번쨰' # ['1번쨰', '2번째', '3번째',....]
- 1번째와 2번째와 같이 문자열로 바뀐 새로운 리스트가 만들어진다.
ex)
my_list = [1,2,3,4,5]
new_list = [(x) (for x in my_list #반복대상) (if x > 3_]

x #[1,2,3,4,5]
x+1 #[2,3,4,5,6]
x*3 #[3,6,9,12,15]
str(x) + '번쨰' # ['1번쨰', '2번째', '3번째',....]


my_list = [1,2,3,4,5]
new_list = [(x) (for x in my_list #반복대상) (if x > 3)]
= 맨 뒤에 if 
- 이렇게 하면 새로운 리스트를 만들때 값이 3보다 큰경우에만
- 그 값을 사용하라는 의미가 있다.

위 의 경우 if 조건문이 없는 경우에 대한 결과 값이고
if 조건문이 있게 되면 x가 3보다 클때 즉 4와 5에 대해서만
새로운 리스트의 요소를 쓰이게 되는데 이렇게 x 값을 다양하게
활용했을 때 실제값은 이렇개 된다

x #[1,2,3,4,5]
x+1 #[2,3,4,5,6]
x*3 #[3,6,9,12,15]
str(x) + '번쨰' # ['1번쨰', '2번째', '3번째',....]


my_list = [1,2,3,4,5]
new_list = [x for x in my_list if x > 3]

풀이 
my_list 에서
3보다 큰 값들만
그대로 사용해서
새로운 리스트로 만들어줘
"""

# 리콜 대상 제품 조회
products = ['JOA-2020', 'JOA-2021', 'SIRO-2021', 'SIRO-2022',]
recall = [] # 리콜 대상 제품 리스트
for p in products:
    if p.startswith('SIRO'): # 제품명이 SIRO 로 시작하는가?
        recall.append(p) # 추가

print(recall)

# List Comprehension 이용하면 코드 해섯시
products = ['JOA-2020', 'JOA-2021', 'SIRO-2021', 'SIRO-2022']
recall = [p for p in products if p.startswith("SIRO")]
print(recall)
# products라는 리스트에서 제품의 이름이 SIRO로 시작하는 경우에만
# 그대로 가져와서 새로운 리스트를 만든다.
"""
- List Comprehension 다양하게 응용하기\
ex)
products = ['JOA-2020', 'JOA-2021', 'SIRO-2021', 'SIRO-2022']
# 모든 모델명 뒤에 SE (Special Edition)을 붙여줘
prod_se = [p + "SE" for p in products]
출력시 ['JOA-2020SE', 'JOA-2021SE', 'SIRO-2021SE', 'SIRO-2022SE',]

# 모든 모델명을 소문자로 바꿔줘
prod_lower = [p.lower() for p in products]
['joa-2020', 'joa-2021', 'siro-2021', 'siro-2022']

# 22년 제품만 뽑는데 뒤에 (최신형)이라는 글자를 붙여줘
prod_new = [p+'(최신형)' for p in products if p.endswith('2022')]
['SIRO-2022(최신형)]
"""

my_list = ['korea', 'English', 'france']
new_list = [x.upper() for x in my_list if 'a' in x]
print(new_list)


################# 함수 ###################
# 어떤 동작을 수행하는 코드들의 묶음
# 여러 곳에서 사용되는 코드는 하나의 함수
"""
함수의 정의 
def 함수():
    수행할 문장


"""
def show_price(): # 함수 정의
    print('커트 가격은 10000원 입니다.')

show_price() # 함수 호출

# 최초의 프로그램
customer1 = '장발'
print(f"{customer1} 고객님")
print('커트가격은 10000원 입니다.')

customer2 = '나수염'
print(f"{customer2} 고객님")
print('커트가격은 10000원 입니다.')

# 함수를 호출하지않으면 출력이 되지 않는다

# 함수 만들어서 적용하기 

def show_prie(): # 함수 정의
    print('커트 가격은 10000원 입니다.')

customer1 = '장발'
print(f"{customer1} 고객님")
show_price() # 만든 함수 호출하기

customer2 = '나수염'
print(f"{customer2} 고객님")
show_price() # 함수호출

###### 전달값 ############

def show_prie(): # 함수 정의
    print('커트 가격은 15000원 입니다.')

customer1 = '나장발'
print(f"{customer1} 고객님")
show_price() # 만든 함수 호출하기

customer2 = '나수염'
print(f"{customer2} 고객님")
show_price() # 함수호출
"""
전달값 => 파라미터 
def 함수명(전달값):
    수행할문장


일일이 하나하나 작성할 필요 없이 
(원본)함수를 만들어서 간편하게 코드를 작성하고

customer1에 


"""
def show_prie(customer): # 함수 정의  ()안에 전달값 설정
    print(f"{customer} 고객님") # print 문 추가
    print('커트 가격은 15000원 입니다.')

# customer1 = '나장발'
# # print(f"{customer1} 고객님")  # 불필요한 코드제거
# show_price(customer1)  # 호출 시 값 전달

# customer2 = '나수염'
# show_price(customer2) # 함수호출


######################### 반환값 ##############################
# 함수 내에서 처리된 결과를 반환
"""
함수
def 함수명(전달값):
    수행할 문장
    return 반환값

def get_price(): # 함수 정의
    retun 15000

price = get_price() # 함수 호출




# """

print("======================================")
def get_price(): # 함수 정의
    return 15000

price = get_price() # 함수 호출
print(f"커트 가격은 {price} 원 입니다. ")

def get_price(is_vip): # True: 단골 손님, False: 일반손님
    if is_vip == True:
        return 10000 # 단골 손님
    else:
        return 15000  #일반손님

price = get_price(True)
print(f"커트 가격은 {price} 원 입니다. ") # 10000

# 보통 반환 값은 하나이지만 여러 개 반환 가능(콤마로 구분, 튜플)
# return 키워드를 통해서 값을 반환하게 되면 그 즉시 함수를 탈출하게 된다.
# 함수 내에서 처리된 결과를 반환하는 키워드는 return 
price1=get_price(True)  # 단골손님
price2=get_price(False) # 일반손님
price3=get_price(False) # 일반손님
price4=get_price(False) # 일반손님
# 기본 값 이용하면 함수의 사용이 굉장히 편리해질수 있다
# 기본 값 이란 전달값 에 기본으로 사용되는 값입니다.
"""
def 함수명(전달값=기본값):
    수행할 문장

"""
print("========================================")
# def get_prive(is_vip=False): # True: 단골손님, False: 일반손님 기본값 설정
#     if is_vip == True:
#         return 10000 # 단골 손님
#     else:
#         return 15000 # 일반 손님
# # 기본값 설정해서 따로 안 넣어줘도 된다
# price1=get_price(True)  # 단골손님
# price2=get_price() # 일반손님
# price3=get_price() # 일반손님
# price4=get_price() # 일반손님

# 기본값은 전달값을 따로 명시하지 않을때 기본으로 설정 되는 값이므로 기본값이 제공되는 함수 호출 시 전달값은 생략 가능 합니다.


# 키워드 값 주기
# price=get_price(reciew=True, is_vip=True) # 키워드 값 주기, 순서는 상관이 없다

def get_prive(is_vip=False): # True: 단골손님, False: 일반손님 기본값 설정
    if is_vip == True:
        return 10000 # 단골 손님
    else:
        return 15000 # 일반 손님

    
    # 가변인자
    # 개수가 바뀔수  있는 인자
    # 전달값 앞에 별하나 찍어주면된다
def visit(today, *customers):
    print(today)
    for customer in customers:
        print(customer)
visit("2022년 6월 10일", "나장발") # 1명 방문
visit("2022년 6월 10일", "나장발", "나수염") # 2명 방문
visit("2022년 6월 10일", "나장발", "나수염", "나김리") # 3명 방문

# 가변인자 주의 사항: 전달값이 많으면 마지막에 한 번만
# 함수의 전닭밧이 여러 개 있다고 할때 가변인자는 마지막에 딱 한 번만 쓸 수 있습니다.
# a는 한개 b는 뒤에 나머지 전부 c는 어떻게 나눠야할지몰라 오류출력
# def my_function(a,*b,*c):

####################### secret() 함수 #########################
# secret 함수에는 message라는 변수를 만들고 어떤 비밀 메세지를 몰래 보관 해둔다.
# 변수의 값을 이런 식으로 출력할 수 있을까?
# ex) print(message)

def secret():
    message = "이건 나만의 비밀"
# 에러가 나는 이유는 지역 변수이기 때문에
# print(message)
# 지역변수란: 함수 내에서 정의된 변수
# 함수내에서만 사용 가능

def secret():
    message = "이건 나만의 비밀" # secret 함수 내에서 정의
    print(message) # 값 출력 가능
    message = "함수 내에서는 자유롭게 수정 가능하다."
""" 지역변수

# def please():
#   print(message) # 절대 안됨
#print(message) # 절대 안됨

함수 밖에서 또는 다른 함수에서 접근 수정 X


def secret():
    message = "이건 나만의 비밀" # secret 함수 내에서 정의
    print(message) # 값 출력 가능
    message = "함수 내에서는 자유롭게 수정 가능하다."

def pleas():
    message = "이렇게 하면 될까?" # secret 함수 내에서 정의
    print(message) # 값 출력 가능
서로 다른 함수이다. 이름만 같을뿐!
"""
######################## 전역변수 ############################
"""
어디서든 사용 할 수 있다.
함수 밖에서 만들면 전역 변수이다.
Ex) 
message = "나느야 전역변수"

print(message) 함수 밖에서나

- 함수 안에서나 사용가능
- 주의 할점
    함수안에서 전역 변수와 똑같은 이름의 message라는 변수에 값을 할당하려고 하면
    전역변수가 아닌 no_secet() 함수 내의 지역 변수로 새로 만들어져서
    출력 값이 달라진다.
    전역 변수 message의 값은 변하지 않는다. 
    함수 내에서 전역 변수의 값을 사용만 할때는 상관 없지만
    값을 직접 수정하려고 할때는 global 키워드를 써서 global message로 선언한다.

이렇게 하면 전역 공간의 변수를 사용하게 되고 만약 전역 공간에 이 변수가 없다면 
message라는 이름의 전역 변수를 만들게된다.

def no_secret():
    print(message) # '나는야 전역 변수'출력

def no_secret():
    global message # 전역 변수 사용하겠음. 없으면 여기서 만들겠음
    # message= "이러면 또 지역 변수"
    print(message) # '나는야 전역 변수'출력
 
"""
print("=====================================================================")
message = "나느야 전역변수"

def no_secret():
    message= "이러면 또 지역 변수"
    print(message) # '나는야 전역 변수'출력

def no_secret():
    global message # 전역 변수 사용하겠음. 없으면 여기서 만들겠음
    # message= "이러면 또 지역 변수"
    print(message) # '나는야 전역 변수'출력

# name = input("예약자분 성함이 어떻게 되나요?")
# print(name)


# num = int(input("총 몇분이세요?"))
# if num > 4:
#     print('죄송하지만 저희 식당은 최대 4분까지만 예약 가능합니다.')

# w (쓰기모드) encodin=utf8 은 글자명
f = open("파일명_list.txt", 'w', encoding='utf8')
f.write('김\n')
f.write('정\n')
f.write('허\n')
f.close()

# 파일 읽어오기 r
f = open("파일명_list.txt", 'r', encoding='utf8')
contents = f.read()
print(contents)

f.close() # 파일 닫기 ★꼭 파일 닫아주어야한다.

# 한줄씩 읽어오기
f = open("파일명_list.txt", 'r', encoding='utf8')
for line in f:
    print(line, end='')
f.close()

# 파일 닫기
# with # 자동으로 파일 닫아준다.
with open("파일명_list.txt", 'w', encoding='utf8')as f:
    f.write('김\n')
    f.write('정\n')
    f.write('허\n')

 # 읽어오기
with open("파일명_list.txt", 'r', encoding='utf8')as f:
    contents = f.read()
    print(contents)

# 파일 입출력할때 with 구문을 쓰는 버릇을 만들자
# with open (파일명,r,인코딩번호)as f:

###################### class #########################
# ex) 블랙박스

# 값 넣기
name= '까망이'
resolution = "FHD"
price = 200000
color = 'black'

"""
클래스는 여러 변수들을 묶어서 한번에 관리 할 수 있다.
클래스 안에 어떤 기능을 하는 함수와 같은 걸 만들어서 동작하도록 할 수 있따.
클래스는 설계도와 설명서를 합친 것으로 이해하면 좋다.
설명에 따라서 블랙박스의 정보도 확인 할 수 있고 설계도에 따라서 블랙박스를 만들고
또 블랙박스가 가진 기능이 어떻게 동작하는지도 알 수 있다.
설계도는 하나만 가지고 있어도 여러 블랙 박스 제품을 만들 수 있다.

- 클래스는 한번만 정의를 해두면 얼마든지 제품을 만들 수 있는데
    이제품들을 object, 우리말로는 객체라고한다.
이 때 각 객체는 instence(인스턴스)라고 표현한다.



class 클래스명:
    정의

블랙박스에 대한 정의 예)
클래스명은 대문자로 시작하는 단어들의 조합으로 만들면 된다

class BlackBox:
    pass



"""
class BlackBox:
    pass

b1 = BlackBox()
b1.name = '까망이'
print(b1.name)
print(isinstance(b1,BlackBox))

############################# __init__ #############################
class BlackBox:
    def __init__(self, name, price):
        self.name = name
        self.price = price

b1 = BlackBox('까망이', 200000)
print(b1.name, b1.price)

b2 = BlackBox('하망이', 100000)
print(b2.name, b2.price)


"""
BlackBox 클래스 설계도 제작은
변수1:name
변수2:price

b1, b2 객체는 각각 이런 형태가 된다.
name:'까망이'
price:200000
name:'하양이'
price:100000

# 참고사항
__init__ 함수는 객체를 생성 할 때 자동으로 실행된다.

__init__ 함수는 일반 함수와 동일하게 def 를 통해 정의하지만
괄호 소 다른 변수들 앞에 self가 들어간다는 점을 참고하자.
"""

############################# 멤버변수 ###########################
"""
class BlackBox:
    def __init__(self, name, price):
        self.name(멤버변수) = name
        self.price(멤버변수) = price
# b1 객체
name:'까망이'
price:200000
nickname: 1호
# b2 객체
name:'하양이'
price:100000


"""

############################ 메소드 ###############################
# 클래스 내에서 선언되는 함수는 __init__ 메소드라고 한다.
class BlackBox:
    def __init__(self, name, price):
        self.name = name
        self.pric = price

    def set_travel_mode(self):# 메소드 만들기
        print("여행 모드 ON")

# b1 = BlackBox('까망이', 200000)
# b1.set_travel_mode()

# 전달값이 필요한 경우
    def set_travel_mode(self,min): # 여행 모드 시간 (분)
        print(str(min) + '분 동안 여행모드 On')


b1 = BlackBox('까망이', 200000)
b1.set_travel_mode(20)

# 메소드는 self를 제외한 전닭밧들을  일반 함수와 같은 방식으로 호출하면 된다.
# 함수를 정의 할때 전달값이 두개면
def add(num1, num2):
    add(1,2) # 함수 호출
# self 객체의 자기 자신을 의미한다.
class BlackBox:
    def __init__(self, name, price):
        self.name = name
        self.pric = price

    def set_travel_mode(self, min):# 메소드 만들기
        print(f"{self.name} {min}분 동안 여행 모두 ON")

b1 = BlackBox('까망이', 200000)
b2 = BlackBox('하망이', 100000)

b1.set_travel_mode(20) # 1번 코드
b2.set_travel_mode(10)

BlackBox.set_travel_mode(b1,20) # 2번 코드

# self 
# 1. 메소드를 정의할 때 처음 전달값은 반드시 self
# 2. 메소드 내에서는 self.name과 같은 형태로 멤버 변수를 사용

####################### 상속 ############################
# 기본 블랙바스 
class BlackBox: # 부모 클래스
    def __init__(self, name, price):
        self.name = name
        self.pric = price
        
# 여행 모드 지원 블랙박스
class TravelBlackBox(BlackBox): # 자식 클래스
    def __init__(self, name, price):
    #     self.name = name
    #     self.pric = price
        super().__init__(self,name,price)
    def set_travel_mode(self,min): # 여행 모드 시간 (분)
        print(str(min) + '분 동안 여행모드 On')

# 겹치는 부분 두클래스와 init()메소드 일치
# 멤버변수도 똑같이 name과 price만 가지고

# 상속
# 부모가 자식에게 물려주는것

#######################  다중 상속 #############################
# 추억용 영상 제작 기능 구현 클래스
class VideoMaker:
    def make(self):
        print('추억용 여행 영상 제작')

# 여행모드 지원 블랙박스
class MailSender:
    def send(self):
        print('메일 발송')

class TravelBlackBox2(BlackBox, VideoMaker, MailSender):
    def __init__(self, name, price, sd):
        # super().__init__(name, price)
        self.sd = sd

    def set_travel_mode(self,min):
        print(str(min) + '분 동안 여행 모드 ON')
# 다중 상속 = 여러 클래스에게 상속받는것

b1 = TravelBlackBox2('하양이', 100000, 64) # name,price, sd
b1.make() 
b1.send() 

# 3개의 클래스로 다중상속을 받았다.
# 기본 블랙박스 & 추억용 영상 제작 기능 구현 클래스 & 메일 발송 기능 구현 클래스
# 위 3개를 합쳐 여행 모드 지원 블랙바스

########################### 메소드 오버라이딩 #########################
class AdvancedTravelBlackBox(TravelBlackBox2):
    def set_travel_mode(self,min):
        print(str(min) + '분 동안 여행 모드 ON')
        self.make()  # 추억용 여행 영상 제작
        self.send()  # 메일 발송

"""
자식 클래스에서 같은 메소드를 새로 정의하지 않으면 
부모 클래스의 메소드를 가져오고

자식클래스에엇 같은 메소드를 새로 정의하면
자식 클래스의 메소드를 쓰게된다

메소드 오버라이딩 메소드를 새로 정의 하는것

"""
b1 = TravelBlackBox2('하양이', 100000, 64)
b1.set_travel_mode(20)

b2 = TravelBlackBox2('초록이', 120000, 64)
b2.set_travel_mode(15)
# 추억용 여행 영상 제작과 메일 발송 까지 수행

# 부모 클래스의 메소드를 자식 클래스에서 새롭게 정의 하는 것은 메소드 오버라이딩
# overriding


###################### pass #######################333 
# pass는 일단 내버려두라는 의미
class BlackBox:
    def __init__(self):
        pass
    
    def record(self):
        pass
    
    def stop(self):
        pass
    
    def format(self): # sd 카드 포맷
        pass        # 에러발생X

# 함수 또는 if wile문 for문에도 사용이 가능하다

################### 예외처리 ########################
"""
예외처리 사용법
try:
    수행문장
except:
    에러 발생 시 수행 문장 # 에러 상황이 발생했을 때만 수행할 문장
else:
    정상 동작 시 수행 문장 # 애러가 발생하지 않았을 때만 수행 할문장
finally:
    마지막으로 수행할 문장  # 에러 여부 상관없이 항상 수행되는 문장

"""
num1 = 3
num2 = 3

try:
    result = num1 /num2 # num1 = 3 num2 = 0 이라고 가정
    print(f"연산 결과는 {result} 입니다.")
except:
    print("에러가 발생 했어요.")

else:
    print("정상 동작했어요.")

finally:
    print("수행종료")


"""
예외처리 구문 4가지 유형의 사용법
# try: 는 항상 except 또는 finally: 와 함께 쌍을 이뤄야 한다.
try:
    수행문장
except:
    에러처리


try:
    수행문장
finally:
    마지막 수행


try:
    수행문장
execpt:
    에러처리
else:
    정상동작


try:
    수행문장
execpt:
    에러처리
else:
    정상동작
finally:
    마지막 수행


"""
########################## 에러 #############################
num1 = 3
num2 = 0

# 해경 방법 새로운 except 구문으로 추가 또는 해당 구문애 예외처리
try:
    result = num1 /num2 # num1 = 3 num2 = 0 이라고 가정
    print(f"연산 결과는 {result} 입니다.")
except ZeroDivisionError:
    print("0으로 나눌 수 없어요.")
except TypeError:
    print("값의 형태가 이상해요.")
except Exception as err: # err은 임의로 설정가능
    print("에러가 발생 했어요.: ", err)
else:
    print("정상 동작했어요.")

finally:
    print("수행종료")

################################# 모듈 #######################################
"""
# 모듈이란 코드들이 작성되어 있는 하나의 파이썬 파일을 의미한다.
# 모듈에는 변수, 함수, 클래스 등이 정의 되어있다.

# 모듈 만들기

# def say():
#     print("참 잘했어요.")

# 새로운 파일에서 이 모듈을 가져다 쓰기 위해서는 2가지 방법이 있다.

1) import 모듈
2) from 모듈 import 변수, 함수 또는 클래스
"""
# 1번째 방법
# 모듈에 전체 포함된 모든 기능 다 쓸수 있도록 하겠다는 의미
import nadocoding.goodjob as goodjob
goodjob.say()

# 모듈중에서 필요한 것들만 say()함수만 가져다 쓰겠다는 의미
# 2번째 방법
from nadocoding.goodjob import say
say()

# random 모듈 함수
import random
my_list = ['가위', '바위', '보']
print(random.choice(my_list))

# 참고 사이트
# 구글에서 list of python modules 검색해서 링크를 통해서 사용법들이 많이 있음
#  모듈에서는 하나의 함수만 사용하기 위해서는 from 모듈 import 함수 

########################### 패키지 #########################
"""
코드들이 작성되어 있는 하나의 파이썬 파일을 모듈이라고 한다.
이런 모듈이 여러개 모인것이 패키지라고 한다.

패키지 = 모듈1 - 하나의 파이썬 파일(.py), 모듈2 - 하나의 파이썬 파일(.py), 모듈 - 하나의 파이썬 파일(.py)3

패키지는 하나의 폴더이고 그안에 여러 모듈들이 존재하는 형태인데

"""
# 1번째 방법
import nadocoding.goodjob
nadocoding.goodjob.say()

# 2번째 방법
from nadocoding import goodbye
goodbye.bye()

# 두개 모듈을 다 갖다 쓰고 싶을 때
from nadocoding import goodjob,goodbye
goodjob.say()
goodbye.bye()


