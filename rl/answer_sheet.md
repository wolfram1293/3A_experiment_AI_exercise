# 演習解答欄

## 演習 3.4.2

作成している行
agent:10
env:8

呼び出されている変数・メソッド

agent:
*.act_and_train(obs, reward, done)
*.act(obs)
*.get_statistics()
*.stop_episode_and_train(obs, reward, done)
*.stop_episode()

env:
*.metadata
*.reset()
*.render(render_mode)
*.step(action)

## 演習 3.4.3

平均step数

train:48.76
test:48.73

## 演習 3.4.7

平均step数

train:7.33
test:6.48

train first 10:17.4
train last 10:5.5

## 演習 3.4.9

平均step数

train:8.57
test:4.0

## 演習 3.4.12

学習episode数を増やす、バッチサイズを上げる、epsを時間変化させるなどの改良の結果、点数は平均142.5点程度に改善した。
