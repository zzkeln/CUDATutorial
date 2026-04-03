# 总结
+ shared memory被分为多个bank，划分的方法是：bank = (addr / 4) % 32
+ bank conflict是shared memory并发访问的效率问题

好，这一块是 CUDA 性能优化里**非常关键但又很容易被低估的点**。你如果把 **warp + memory coalescing + bank conflict** 这三件事吃透，基本就具备写高性能 kernel 的核心能力了。

我给你从**硬件结构 → 访问规则 → 冲突本质 → 例子 → 计算方法 → 优化技巧 → 常见误区**系统讲清楚。

---

# 一、什么是 bank？

在 CUDA 中说的 **bank**，指的是：

> **shared memory 被划分成多个可以并行访问的“内存通道”**

---

## 1️⃣ shared memory 结构（重点）

* shared memory ≠ 一整块连续内存直接访问
* 它被划分成：

```text
32 个 bank（大多数架构）
```

👉 每个 bank：

* 一次只能服务 **一个地址访问**
* 但 32 个 bank 可以并行工作

---

## 2️⃣ 类比理解

你可以把 shared memory 想成：

```text
一个银行大厅（shared memory）
有 32 个窗口（bank）
每个窗口一次只能服务一个人
```

---

# 二、bank 是如何映射地址的？

这是理解 conflict 的核心。

---

## 映射规则（经典模型）

假设：

* 每个 bank 宽度 = 4 字节（int / float）
* bank 数量 = 32

👉 那么：

```text
bank_id = (address / 4) % 32
```

---

## 举例：

```cpp
shared[0] → bank 0
shared[1] → bank 1
...
shared[31] → bank 31
shared[32] → bank 0（循环）
```

👉 **bank 是按地址轮转的**

---

# 三、什么是 bank conflict？

> **当一个 warp 内多个线程同时访问同一个 bank 的不同地址时，就发生 bank conflict。**

---

## 情况分类（非常重要）

---

## ✅ 情况1：无冲突（最理想）

```cpp
shared[threadIdx.x]
```

👉 每个线程访问不同 bank：

| thread | addr | bank |
| ------ | ---- | ---- |
| 0      | 0    | 0    |
| 1      | 1    | 1    |
| ...    | ...  | ...  |
| 31     | 31   | 31   |

✔ 完全并行（1 cycle）

---

## ❌ 情况2：bank conflict

```cpp
shared[threadIdx.x * 2]
```

👉 bank 映射：

| thread | addr | bank |
| ------ | ---- | ---- |
| 0      | 0    | 0    |
| 1      | 2    | 2    |
| 16     | 32   | 0 ❗  |

👉 多个线程访问同一个 bank

结果：

> **访问被拆成多次执行（串行）**

---

## ❗ 情况3：broadcast（特殊优化）

```cpp
shared[0]
```

👉 所有线程访问同一个地址

✔ GPU 优化为：

> **broadcast（一次完成）**

👉 不算 conflict！

---

# 四、bank conflict 的本质

核心一句话：

> **bank 是并行资源，冲突 = 资源竞争 → 串行化**

---

## 执行方式：

如果：

* 有 N 个线程访问同一个 bank

👉 实际执行：

```text
分 N 次访问完成
```

---

## 性能影响：

```text
访问时间 ≈ conflict degree（冲突度）
```

---

# 五、如何计算 conflict（非常实用）

## 步骤：

1. 计算每个线程访问的 address
2. 映射到 bank：

```text
bank = (addr / 4) % 32
```

3. 看同一个 bank 有多少线程访问

---

## 示例分析

```cpp
shared[threadIdx.x * stride]
```

---

### 情况1：stride = 1

👉 无冲突 ✅

---

### 情况2：stride = 2

👉 每个 bank 被 2 个线程访问

👉 **2-way conflict**

---

### 情况3：stride = 32

👉 所有线程访问：

```text
bank = (threadIdx.x * 32) % 32 = 0
```

👉 💥 32-way conflict（最坏）

---

# 六、经典问题：矩阵转置（必考）

---

## ❌ naive 写法（有 conflict）

```cpp
tile[threadIdx.x][threadIdx.y]
```

读取：

```cpp
tile[threadIdx.y][threadIdx.x]
```

👉 访问列 → stride = 32

👉 💥 32-way conflict

---

## ✅ 优化：padding

```cpp
tile[32][33]
```

👉 多加一列

---

### 为什么有效？

原来：

```text
stride = 32 → bank 冲突
```

现在：

```text
stride = 33
bank = (x * 33) % 32
```

👉 每个线程落在不同 bank

✔ 无冲突！

---

# 七、bank conflict vs global memory coalescing

很多人会混淆：

| 特性   | shared memory | global memory |
| ---- | ------------- | ------------- |
| 关注点  | bank conflict | coalescing    |
| 问题本质 | bank 冲突       | 访问合并          |
| 影响   | latency       | bandwidth     |

---

# 八、什么时候容易出现 bank conflict？

---

## ❌ 1. stride 访问

```cpp
shared[threadIdx.x * k]
```

---

## ❌ 2. 转置访问

```cpp
tile[col][row]
```

---

## ❌ 3. 不规则访问

```cpp
shared[index[threadIdx.x]]
```

---

# 九、如何避免 bank conflict？

---

## ✅ 方法1：padding（最常用）

```cpp
__shared__ float tile[32][33];
```

---

## ✅ 方法2：改变访问模式

👉 尽量让：

```text
threadIdx.x → 连续地址
```

---

## ✅ 方法3：使用 vector 类型

```cpp
float4
```

👉 改变访问粒度

---

## ✅ 方法4：warp-level shuffle（高级）

```cpp
__shfl_sync
```

👉 避开 shared memory

---

# 十、一个直观比喻

想象：

```text
32 个线程 = 32 个人
32 个 bank = 32 个窗口
```

---

### 无冲突：

```text
每人去不同窗口 → 一次完成
```

---

### 有冲突：

```text
10 个人挤一个窗口 → 排队
```

---

### broadcast：

```text
所有人问同一个问题 → 一次广播回答
```

---

# 十一、面试级总结

> **bank conflict 是 shared memory 中多个线程访问同一 bank 的不同地址导致的访问串行化问题，其性能损失与冲突度成正比，可以通过 padding、调整访问模式等方式避免。**

---

# 十二、再往深一点（进阶方向）

如果你想继续深入（很适合你这种已经在看 CUDA 底层的）：

我可以帮你讲：

1. **不同架构 bank width（4B vs 8B）影响**
2. **ldmatrix / tensor core 如何避免 bank conflict**
3. **Nsight Compute 如何看 shared memory bank conflict**
4. **实际 kernel（如 GEMM）里如何彻底消除 conflict**

你可以直接说：“我想看 GEMM 里的 bank conflict 优化”，我可以带你到工业级实现（接近 cuBLAS 思路）。
