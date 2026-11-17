# 作シミュ（作者がシミュレーションしてみた）の計算

YouTube「作者がシミュレーションしてみた（作シミュ）」の動画で使った計算のコード。動画で見せたシミュレーションの結果は、すべてここで計算したもの。

| フォルダ | 回 | 元にした研究 |
|---|---|---|
| [huddle/](huddle/) | #01 自分勝手なペンギンほど公平になる？（コウテイペンギンのハドル） | Waters, Blanchette, Kim (2012) "Modeling Huddling Penguins", PLoS ONE 7(11): e50277 |

[uv](https://docs.astral.sh/uv/) で動かす。

```
cd huddle
uv run python run.py --n 100 --pe 10 --noise 0 --steps 400 --name n100_pe10
uv run python plot.py n100_pe10
```
