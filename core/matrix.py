"""矩阵运算：multiply、determinant、inverse、transpose、rank、solve。"""

from mpmath import mp

from .errors import MathError
from .registry import MATRIX_OPERATIONS
from .statistics import valid_number

_MAX_SOLVE_SIZE = 50
# 有限浮点数尺度跨度约 632 个十进制量级，LU 求解需要额外精度。
_SOLVE_DPS = 700


def matrix_operation(matrices: list, operation: str):
    """对 1 个（或乘法/求解所需的 2 个）矩阵执行指定运算。"""
    if operation not in MATRIX_OPERATIONS:
        raise MathError(
            f"未知矩阵运算: {operation}",
            "unknown_operation",
            hint=f"可用运算: {', '.join(MATRIX_OPERATIONS)}",
            valid_options=list(MATRIX_OPERATIONS),
            example='matrix_operation([[[1, 2], [3, 4]]], "determinant")',
        )
    if not isinstance(matrices, list) or len(matrices) == 0:
        raise MathError(
            "matrices 必须是非空矩阵列表",
            "invalid_argument",
            hint="multiply 和 solve 需要两个矩阵，其余运算需要一个矩阵",
        )
    if operation == "solve":
        if len(matrices) != 2:
            raise MathError(
                "solve 需要恰好两个矩阵",
                "invalid_argument",
                hint='matrix_operation([A, b], "solve")',
            )
        return _solve(
            _validate(matrices[0], max_size=_MAX_SOLVE_SIZE, convert_to_float=False),
            _validate(matrices[1], max_size=_MAX_SOLVE_SIZE, convert_to_float=False),
        )

    if operation == "multiply":
        if len(matrices) != 2:
            raise MathError(
                "矩阵乘法需要恰好两个矩阵",
                "invalid_argument",
                hint='matrix_operation([A, B], "multiply")',
            )
        return _multiply(_validate(matrices[0]), _validate(matrices[1]))

    if len(matrices) != 1:
        raise MathError(
            f"{operation} 只接受一个矩阵",
            "invalid_argument",
            hint=f'.matrix_operation([A], "{operation}")',
        )
    matrix = _validate(matrices[0])
    if operation == "determinant":
        return _determinant(matrix)
    if operation == "inverse":
        return _inverse(matrix)
    if operation == "transpose":
        return _transpose(matrix)
    return _rank(matrix)


def _validate(matrix, *, max_size=None, convert_to_float=True) -> list[list[int | float]]:
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise MathError(
            "矩阵必须是非空列表的列表",
            "invalid_argument",
            hint="矩阵形如 [[1, 2], [3, 4]]",
        )
    if max_size is not None and (
        len(matrix) > max_size
        or any(isinstance(row, list) and len(row) > max_size for row in matrix)
    ):
        raise MathError(
            f"矩阵规模超过上限 {max_size}x{max_size}",
            "invalid_argument",
            hint=f"solve 最多支持 {max_size} 阶方阵",
        )
    if not all(isinstance(row, list) and len(row) > 0 for row in matrix):
        raise MathError(
            "矩阵每行必须是非空列表",
            "invalid_argument",
            hint="矩阵形如 [[1, 2], [3, 4]]",
        )
    cols = len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise MathError(
            "矩阵行长度不一致",
            "invalid_argument",
            hint="矩阵必须是规整的二维列表",
        )
    for row in matrix:
        for entry in row:
            if not valid_number(entry):
                raise MathError(
                    "矩阵元素必须是有限数字",
                    "invalid_argument",
                    hint="不接受字符串、布尔值、NaN 或无穷大",
                )
    if not convert_to_float:
        return matrix
    return [[float(entry) for entry in row] for row in matrix]


def _multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    a_cols, b_rows = len(a[0]), len(b)
    if a_cols != b_rows:
        raise MathError(
            f"矩阵维度不匹配: {len(a)}x{a_cols} 乘 {b_rows}x{len(b[0])}",
            "matrix_dimension",
            hint="第一个矩阵的列数必须等于第二个矩阵的行数",
        )
    b_cols = len(b[0])
    return [
        [
            sum(a[i][k] * b[k][j] for k in range(a_cols))
            for j in range(b_cols)
        ]
        for i in range(len(a))
    ]


def _determinant(matrix: list[list[float]]) -> float:
    rows, cols = len(matrix), len(matrix[0])
    if rows != cols:
        raise MathError(
            f"行列式要求方阵，收到 {rows}x{cols}",
            "matrix_dimension",
            hint="只有方阵才有行列式",
        )
    echelon, swaps, _ = _echelon(matrix)
    det = -1.0 if swaps % 2 else 1.0
    for i in range(rows):
        det *= echelon[i][i]
    return det


def _inverse(matrix: list[list[float]]) -> list[list[float]]:
    n = len(matrix)
    if n != len(matrix[0]):
        raise MathError(
            f"求逆要求方阵，收到 {n}x{len(matrix[0])}",
            "matrix_dimension",
            hint="只有方阵可求逆",
        )
    scale = max(abs(entry) for row in matrix for entry in row) or 1.0
    threshold = 1e-12 * scale
    aug = [
        [float(entry) for entry in row]
        + [1.0 if i == j else 0.0 for j in range(n)]
        for i, row in enumerate(matrix)
    ]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) <= threshold:
            raise MathError(
                "矩阵奇异（不可逆）",
                "singular_matrix",
                hint="行列式为 0 的矩阵没有逆矩阵",
            )
        aug[col], aug[pivot] = aug[pivot], aug[col]
        diag = aug[col][col]
        aug[col] = [entry / diag for entry in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [
                entry - factor * pivot_entry
                for entry, pivot_entry in zip(aug[r], aug[col])
            ]
    return [row[n:] for row in aug]


def _transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*matrix)]


def _rank(matrix: list[list[float]]) -> int:
    _, _, pivots = _echelon(matrix)
    return pivots


def _echelon(matrix: list[list[float]]):
    """部分主元高斯消元，返回 (行阶梯矩阵, 行交换次数, 主元数)。"""
    rows, cols = len(matrix), len(matrix[0])
    grid = [row[:] for row in matrix]
    scale = max(abs(entry) for row in grid for entry in row) or 1.0
    threshold = 1e-12 * scale
    swaps = 0
    pivots = 0
    for col in range(cols):
        if pivots >= rows:
            break
        best = max(range(pivots, rows), key=lambda r: abs(grid[r][col]))
        if abs(grid[best][col]) <= threshold:
            continue
        if best != pivots:
            grid[pivots], grid[best] = grid[best], grid[pivots]
            swaps += 1
        pivot_value = grid[pivots][col]
        for r in range(pivots + 1, rows):
            factor = grid[r][col] / pivot_value
            grid[r] = [
                entry - factor * pivot_entry
                for entry, pivot_entry in zip(grid[r], grid[pivots])
            ]
        pivots += 1
    return grid, swaps, pivots


def _solve(a, b):
    """高精度 LU 求解线性方程组 Ax=b，输入已校验。"""
    n = len(a)
    if len(a[0]) != n:
        raise MathError(
            "A 必须是方阵",
            "matrix_dimension",
            hint=f"solve 需要 n x n 方阵，当前为 {n}x{len(a[0])}",
        )
    if len(b) != n or any(len(row) != 1 for row in b):
        raise MathError(
            "b 必须是 n 行 1 列的列向量",
            "matrix_dimension",
            hint=f"A 为 {n}x{n}，b 应为 [[b1], ..., [b{n}]]",
        )
    ctx = mp.clone()
    ctx.dps = _SOLVE_DPS
    try:
        solution = ctx.lu_solve(a, b)
    except ZeroDivisionError as exc:
        raise MathError(
            "矩阵数值奇异，无法求解",
            "singular_matrix",
            hint="A 可能奇异或病态，当前数值精度下无法求解",
        ) from exc
    result = [[float(solution[i])] for i in range(n)]
    for row in result:
        for value in row:
            if not valid_number(value):
                raise MathError(
                    "解超出有限数值范围",
                    "domain_error",
                    hint="解不能是 NaN 或 Infinity",
                )
    return result
