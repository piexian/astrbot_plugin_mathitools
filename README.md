# 数学工具集

AstrBot 4.29.1+ 的 10 个数学 FunctionTool，基于 mpmath / asteval。

> [!NOTE]
> 需要 AstrBot >=4.29.1,<5；工具名从 expression/compare 改为 math_eval/math_compare。

## 快速开始

1. 放入 AstrBot 插件目录
2. 安装 requirements.txt
3. 在配置里设置 dps（默认 20）

## 功能一览

| 工具名 | 用途 |
|---|---|
| math_eval | 求值数学表达式，支持 73 函数、7 常量、复数与列表，^ 按乘方处理 |
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

表达式经 AST 节点白名单校验，禁止 import/属性访问/字符串逃逸；限制长度 1000、AST 深度 10、列表 10000 元素、指数 10000、阶乘 300、单次超时 15s、结果字符串 10000 字符；符号表清空后只注入注册表；精度按调用隔离（mp.workdps），不写全局 mpmath 配置；无网络、无文件写入、无子进程。

## 项目结构

```text
main.py          # 插件入口：注册 10 个工具
core/            # 纯计算层：注册表、安全校验、求值、单位、统计、矩阵、数论、金融
schema.py        # schema 辅助，required 显式
result.py        # JSON 字符串结果封装
math_tools.py    # 10 个 FunctionTool
tests/           # 离线单元测试
.github/workflows/  # Tests（Python 3.12）与 CodeQL Advanced
```

## 二次开发

- 环境：Python 3.12（CI）/ 3.13（本地），安装 requirements.txt
- 测试：`python -B -m unittest discover -s tests -t . -v`
- 新增函数/常量：改 `core/registry.py`，并补 `core/evaluator.py` 符号表与 `math_functions` 输出
- 新增工具：改根目录 `math_tools.py`，在 `main.py` 注册，schema 显式 required
- 错误契约：JSON 字符串，ok/error_code/hint/example/valid_options；失败用 `from astrbot.api import logger` 的 `logger.error`
- 文档同步：改工具/函数/配置时同步 README、CHANGELOG.md、metadata.yaml、_conf_schema.json
- 许可证：AGPL-3.0-or-later，修改与分发遵循 LICENSE

## 开发验证

`python -B -m unittest discover -s tests -t . -v`

离线测试使用最小 AstrBot 桩，不替代真实实例加载。

## 更新日志

详见 CHANGELOG.md。

## 相关链接

- [AstrBot 文档](https://astrbot.app)
- [AstrBot 插件开发文档](https://astrbot.app/dev/)
- [本仓库 Issues](https://github.com/piexian/astrbot_plugin_mathitools/issues)

## 许可

AGPL-3.0-or-later
