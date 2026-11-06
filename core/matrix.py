"""矩阵运算：multiply、determinant、inverse、transpose、rank。"""

from .errors import MathError
from .registry import MATRIX_OPERATIONS
from .statistics import valid_number


def matrix_operation(matrices: list, operation: str):
    """对 1 个（或乘法所需的 2 个）矩阵执行指定运算。"""
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
            hint="multiply 需要两个矩阵，其余运算需要一个矩阵",
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


def _validate(matrix) -> list[list[float]]:
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise MathError(
            "矩阵必须是非空列表的列表",
            "invalid_argument",
            hint="矩阵形如 [[1, 2], [3, 4]]",
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
