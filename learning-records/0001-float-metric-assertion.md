# 浮点指标测试要使用近似比较

今天确认了一个常见测试问题：`accuracy` 在 PyTorch 中经过 float32 tensor 计算后，可能返回 `0.6666666865`，而 Python 的 `2 / 3` 是另一种浮点近似。两者数学意义相同，但不能用 `==` 做严格比较；测试应使用 `pytest.approx` 这样的近似断言。

这条记录会影响后续所有 metric、loss、retrieval score 的测试写法。
