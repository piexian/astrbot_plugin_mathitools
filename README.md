# 数学工具集

基于 [NekoiMeiov/astrbot_plugin_mathitools](https://github.com/NekoiMeiov/astrbot_plugin_mathitools) 二次开发，为 AstrBot 提供表达式求值、单位换算、统计、矩阵、数论与金融计算工具。

> [!NOTE]
> 运行要求：AstrBot >=4.29.1,<5。

## 快速开始

1. 在插件市场搜索「数学工具集」安装，或从链接安装 `https://github.com/piexian/astrbot_plugin_mathitools`。
2. 在插件配置中设置 `dps`（默认 20）。
3. 直接让机器人计算，例如「算一下 sqrt(2)」。

## 功能一览

| 工具名 | 用途 |
|---|---|
| math_eval | 求值数学表达式，支持 78 函数、7 常量、复数与列表，^ 按乘方处理 |
| math_compare | 比较两个实数表达式的大小 |
| math_batch | 批量求值，单项失败落在该项，不中断整批 |
| math_functions | 列出函数、常量、单位、别名与各类 operation 清单 |
| math_statistics | 统计运算（10 种） |
| math_matrix | 矩阵运算（multiply/determinant/inverse/transpose/rank） |
| math_units | 单位换算（15 类 158 单位 + 153 别名，温度为仿射换算） |
| math_nl_units | 自然语言单位换算，如 "5 km to miles" |
| math_number_theory | 数论运算（is_prime/prime_factors/divisors/totient） |
| math_finance | 金融计算（百分比/小费/利息/贷款/分摊/税费等 8 种） |

## 配置说明

| 配置项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| dps | int | 20 | 计算精度，范围 1..200 |

## 安全说明

表达式使用 AST 白名单求值，限制长度、嵌套、指数、超时与结果大小；计算不访问网络、文件或子进程。
聚合与舍入函数 sum/prod/min/max/round 与 Python 3.12 语义一致，round 为半偶数舍入，ndigits 取 -10000..10000；列表按嵌套元素总数限 10000，prod 中间值与整数输出限 10000 位，总输出最多 10000 字符。

## 项目结构
```text
main.py                 # 插件入口
core/                   # 计算层
schema.py               # 工具参数定义
result.py               # 结果封装
math_tools.py           # 数学工具
tests/                  # 测试
.github/workflows/      # 自动化检查
README.md
CHANGELOG.md
LICENSE
metadata.yaml
_conf_schema.json
requirements.txt
.gitignore
```

## 更新日志

详见 CHANGELOG.md。

## 待办事项

- [ ] 符号数学
- [ ] 最小二乘、特征值、SVD
- [ ] 还款明细、NPV、IRR
- [ ] 复合单位与中文单位名

## 相关链接

- [AstrBot 文档](https://docs.astrbot.app/)
- [上游仓库](https://github.com/NekoiMeiov/astrbot_plugin_mathitools)
- [本仓库 Issues](https://github.com/piexian/astrbot_plugin_mathitools/issues)

## 许可

AGPL-3.0-or-later
