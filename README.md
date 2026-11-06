# 数学工具集

基于 [NekoiMeiov/astrbot_plugin_mathitools](https://github.com/NekoiMeiov/astrbot_plugin_mathitools) 二次开发，为 AstrBot 提供表达式求值、单位换算、统计、矩阵、数值求根与微积分、概率分布、数论与金融计算工具。

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
| math_statistics | 统计运算（15 种，支持加权平均、百分位与协方差/相关/回归） |
| math_matrix | 矩阵运算（multiply/determinant/inverse/transpose/rank/solve） |
| math_units | 单位换算（15 类 158 单位 + 153 别名，温度为仿射换算） |
| math_nl_units | 自然语言单位换算，如 "5 km to miles" |
| math_number_theory | 数论运算（is_prime/prime_factors/divisors/totient） |
| math_finance | 金融计算（百分比/小费/利息/贷款/分摊/税费等 8 种） |
| math_numerical | 数值求根/微分/定积分/概率分布（findroot / polyroots / differentiate / integrate / normal_* / binomial_* / poisson_*，13 项运算） |

统计扩展：weighted_mean 通过 weights 传入权重（同长、可为负但和非零）；covariance、correlation、linear_regression 通过 data2 传入第二组数据，data 为 x、data2 为 y。covariance 为样本协方差（n-1），correlation 为 Pearson 相关，linear_regression 返回 slope/intercept。percentile 的 p 取 0..100（含端点）线性插值。每组数据/权重至多 10000 个有限数。

矩阵扩展：solve 通过 matrices=[A, b] 传入方阵 A 与列矩阵 b，求解 Ax=b；A 最多 50 阶，返回解的列矩阵。如 operation="solve"、matrices=[[[2,1],[1,-1]],[[5],[1]]] 返回 [[2.0],[1.0]]。

数值求根：findroot 传 expression/a/b，固定自变量 x，在连续区间二分，端点异号或端点为根；polyroots 传降幂系数（最多 20 次），结果为字符串/字符串数组。dps 取 1..200。重根或病态问题可报 no_convergence。如 operation='findroot', expression='x^2-2', a=1, b=2。

微分与定积分：differentiate 传 expression/x/n（n 默认 1，上限 5）；integrate 传 expression/a/b，允许反向与等端点，points 最多 32 个严格递增的内部拆分点。被积函数需有限连续且足够光滑，已知窄峰/折点用 points 拆分，求值次数用尽或精度不足报 no_convergence。dps 取 1..200，结果以字符串返回。

概率分布：normal_pdf/cdf/ppf 传 x 或 q，mu/sigma 默认 0/1 且 sigma>0；binomial_pmf/cdf/ppf 传 n/p 与 k 或 q；poisson_pmf/cdf/ppf 传 rate 与 k 或 q。n 与 rate 上限 10000，|k| 至多 100000，p/q 取 0..1。离散 ppf 为 CDF>=q 的最小非负整数且 q=0 返回 0；正态 q=0/1 与非退化泊松 q=1 报 domain_error。结果以字符串返回，dps 取 1..200。

## 配置说明

| 配置项 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| dps | int | 20 | 计算精度，范围 1..200 |


## 项目结构
```text
astrbot_plugin_mathitools/
├── core/                      # 计算层
│   ├── __init__.py
│   ├── _numerical_worker.py    # 受限工作进程
│   ├── errors.py               # 错误契约
│   ├── evaluator.py            # 表达式求值
│   ├── finance.py              # 金融计算
│   ├── matrix.py               # 矩阵运算
│   ├── number_theory.py        # 数论
│   ├── numerical.py            # 求根、微积分与概率分布
│   ├── probability.py          # 概率分布
│   ├── registry.py             # 函数与运算清单
│   ├── security.py             # AST 安全校验
│   ├── statistics.py           # 统计
│   └── units.py                # 单位换算
├── tests/                     # 回归测试
│   ├── __init__.py
│   ├── test_core.py
│   ├── test_logger_import.py
│   ├── test_math_ops.py
│   ├── test_numerical.py
│   ├── test_probability.py
│   ├── test_tools.py
│   └── test_units.py
├── benchmarks/                 # 三平台原生预检
│   ├── run.py                  # 父调度汇总
│   ├── native_probe.py         # 内存/CPU 限额探针
│   └── astrbot_probe.py        # 真实 SDK 工具调用
├── .github/
│   └── workflows/
│       ├── codeql.yml           # 安全扫描
│       ├── benchmark.yml       # 三平台原生预检
│       └── tests.yml            # 单元测试
├── main.py                    # 插件入口
├── math_tools.py              # FunctionTool 接口
├── schema.py                  # 工具参数
├── result.py                  # 结果封装
├── _conf_schema.json          # 插件配置
├── metadata.yaml              # 插件元数据
├── requirements.txt           # 运行依赖
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .gitignore
```

## 更新日志

详见 [CHANGELOG.md](CHANGELOG.md)。

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

AGPL-3.0 License

<div align="center">

**如果这个插件对你有帮助，请给个 ⭐ Star 支持一下！**

</div>
