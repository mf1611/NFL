# NFL kaggle competition

## Memo
- 評価指標  
Continuous Ranked Probability Score (CRPS)
各PlayIdごとに、どのくらいヤードを進んだか、失ったかのそれぞれの確率を出す
--> それを、-99から99までの累積分布として表す（-99は99ヤード後ろに進む確率を意味する、-98は-98までの合計確率（-99と-98の合計）がのる形）


- 以前のコンペの図が分かりやすい
https://www.kaggle.com/c/second-annual-data-science-bowl/overview/evaluation

各プレーごとに、
予測した累積分布と、実績のヤード数がある
--> 実績値を閾値とするステップ関数と、累積分布との差の2乗の合計（2乗でなく単に差の合計なら、単純な積分で面積になる）