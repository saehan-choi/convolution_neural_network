import matplotlib.pyplot as plt
from torch._C import LockingLogger
from wandb.proto.wandb_telemetry_pb2 import Labels





f = open('C:\m_results\holy.txt','r')
lines = f.readlines()
num = 0


# ★★★★★★★★★★★★★★★밑에 9를 자동적으로 받아올수 있게끔 해야함★★★★★★★★★★★★★★★
# 이렇게 하면 많이 성공한거
for i in range(9):
    locals()[f'loss_{i}'] = []
    locals()[f'lr_{i}'] = []
    locals()[f'batch_size_{i}'] = []


for line in lines:
    line = line.split()
    print(line)
    locals()[f'loss_{num}'].append(float(line[1]))
    if line[3] == '9':
        locals()[f'lr_{num}'].append(line[5])
        locals()[f'batch_size_{num}'].append(line[-1])
        num+=1
        
# print(lr_0)

    # split하고나면 배열형태가 되기때문에 배열을 잡아줘야 split 다시 할 수 있음


def plt_setting(amount_of_loss):
    # f'y{i}'이거 loss_0, loss_1, loss_2 이런식으로 바꿔야함 들어올때 ㅎㅎ
    for i in range(amount_of_loss):
        # lr = lr_0
        lr = globals()[f'lr_{i}']
        bt_size = globals()[f'batch_size_{i}']
        plt.plot(globals()[f'loss_{i}'],label = f'lr:{float(lr[0])},b_size:{int(bt_size[0])}')

        # 문자열로 된 변수명을 다시 지역
        # 변수명으로 전환
        # ★★★★★ 전역변수로 하고싶다면 locals -> globals로 변환 ★★★★★
        # 어차피 x축이 epochs일거면 x축의 설정은 필요없음
        # x축을 설정해버리면 x,y가 1대1대응이 되어야하므로 오히려 해가됨

    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.legend()
    plt.show()


# 근데 LOSS가 빨리떨어지는것이 꼭 좋은 HYPERPARAMERTERS라고 할수는 없는게, 
# VALIDATION SET 에 OVERFITTING 되는것일수도 있기때문.
# 다만 validation 에 최적화하는것은 무조건 가능


plt_setting(8)
f.close()
