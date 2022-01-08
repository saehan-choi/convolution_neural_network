import matplotlib.pyplot as plt

x = [1,2,3,4,5,6,7,8,9,10,11]
y0 = [2,3,4,5,5,1,2,7,1,1,0]
y1 = [2,7,1,3,16,7,2,4,7,8,4]
y2 = [2,7,1,5,1,2,6,1,3,8,4]
y3 = [2,7,1,5,2,2,6,1,3,8,4]
y4 = [2,7,1,5,6,2,6,1,3,8,4]
y5 = [2,7,1,2,7,1,3,16,3,8,4]
y6 = [2,7,1,5,1,2,2,7,1,3,16]
y7 = [2,7,8,9,10,11,6,1,3,8,4]
y8 = [2,7,1,5,7,8,9,10,11,8,4]

for i in range(9):
    plt.plot(x,locals()[f'y{i}'],label = f'test{i+1}')
    # 문자열로 된 변수명을 다시 지역
    # 변수명으로 전환
    # 전역변수로 하고싶다면 locals -> globals로 변환

plt.xlabel('epochs')
plt.ylabel('loss')

plt.legend()
plt.show()


# 근데 LOSS가 빨리떨어지는것이 꼭 좋은 HYPERPARAMERTERS라고 할수는 없는게, 
# VALIDATION SET 에 OVERFITTING 되는것일수도 있기때문.