这一块和你刚刚问的 **warp / bank conflict** 是同一层级的核心问题，只不过：

> **bank conflict 是 shared memory 的“并发访问冲突问题”**
> **coalescing 是 global memory 的“带宽利用效率问题”**

我给你系统讲清楚，从**直觉 → 硬件行为 → 规则 → 示例 → 常见坑 → 优化套路**。

---

# 一、什么是 global memory coalescing？

一句话定义：

> **warp 内多个线程的内存访问，被合并（coalesce）成尽可能少的内存事务（memory transactions）。**

---

## 直觉理解

一个 warp（32 个线程）：

```cpp
data[threadIdx.x]
```

👉 理想情况：

* 32 个线程访问连续地址
* GPU：

```text
一次内存请求 → 拿回一整块数据（cache line）
```

✔ 这就叫 **coalesced access**

---

## 反例：

```cpp
data[threadIdx.x * 100]
```

👉 每个线程访问远离的地址：

```text
32 个线程 → 32 次独立内存访问
```

❌ 完全不 coalesced

---

# 二、底层到底发生了什么？

---

## 1️⃣ global memory 访问单位

GPU 访问 global memory 时：

* 不是按“一个线程一个请求”
* 而是：

```text
按 cache line / segment（通常 32B / 64B / 128B）
```

👉 一次 transaction 会加载一整块

---

## 2️⃣ warp 的访问会被“合并分析”

GPU 会做：

```text
收集 warp 内所有线程的访问地址
↓
按地址分组
↓
生成最少数量的 memory transactions
```

---

# 三、coalescing 的核心目标

> **让 32 个线程的访问尽可能落在同一或尽量少的内存块中**

---

# 四、最重要的规则（必须记住）

---

## ✅ 理想访问模式

```cpp
data[base + threadIdx.x]
```

👉 特点：

* 连续访问
* 对齐（aligned）

✔ 最优 coalescing

---

## ❌ 不理想

```cpp
data[threadIdx.x * stride]
```

* stride 越大 → 越差

---

# 五、通过例子彻底理解

---

## 示例1：完美 coalescing

```cpp
int idx = threadIdx.x;
float x = data[idx];
```

假设：

* 每个 float = 4B
* warp = 32 threads

👉 总访问：

```text
32 × 4B = 128B
```

👉 GPU：

```text
1 次 128B transaction
```

✔ 完美！

---

## 示例2：stride = 2

```cpp
data[threadIdx.x * 2]
```

访问：

```text
0, 2, 4, 6, ...
```

👉 地址间隔变大

👉 可能需要：

```text
2 次 transaction
```

---

## 示例3：stride = 32（最坏）

```cpp
data[threadIdx.x * 32]
```

👉 每个线程落在不同 cache line

👉 结果：

```text
32 次 transaction 💥
```

---

# 六、alignment（对齐）问题

---

## 为什么对齐重要？

假设：

```cpp
data[threadIdx.x + 1]
```

👉 warp 访问：

```text
从 address = 4B 开始（错位）
```

👉 结果：

* 本来 1 次 transaction
* 现在变成 2 次 ❗

---

## 规则：

> **warp 访问最好按 128B 对齐**

---

# 七、结构体访问问题（常见坑）

---

## ❌ AoS（Array of Structs）

```cpp
struct Point {
    float x, y, z;
};

Point data[N];
```

访问：

```cpp
data[idx].x
```

👉 实际访问：

```text
x x x x x x ...
但内存是 xyz xyz xyz
```

👉 ❌ 不连续 → coalescing 差

---

## ✅ SoA（Structure of Arrays）

```cpp
float x[N], y[N], z[N];
```

👉 连续访问

✔ coalesced

---

# 八、coalescing vs cache

---

## 有 cache 后还重要吗？

👉 仍然非常重要！

原因：

* GPU cache 较小
* 带宽是瓶颈
* coalescing 决定：

```text
带宽利用率
```

---

# 九、和 bank conflict 的本质区别

| 对比   | coalescing    | bank conflict |
| ---- | ------------- | ------------- |
| 内存类型 | global memory | shared memory |
| 本质   | 合并请求          | 避免冲突          |
| 影响   | 带宽利用          | 访问延迟          |
| 单位   | warp          | warp          |

---

# 十、如何判断 coalescing 好坏？

---

## 一个简单指标：

```text
实际 transaction 数 / 理想 transaction 数
```

---

## Nsight Compute 中：

可以看：

* Global Memory Load Efficiency
* DRAM Throughput

---

# 十一、优化技巧（实战）

---

## ✅ 1. 保证连续访问

```cpp
data[threadIdx.x]
```

---

## ✅ 2. 避免 stride

如果必须 stride：

👉 用 shared memory 重排

---

## ✅ 3. 使用 SoA 替代 AoS

---

## ✅ 4. 对齐内存

```cpp
__align__(16)
```

或：

```cpp
cudaMalloc → 默认对齐
```

---

## ✅ 5. vectorized load

```cpp
float4 val = ((float4*)data)[idx];
```

👉 一次读 16B

---

## ✅ 6. 使用 shared memory 做“中转”

经典套路：

```text
global（不规则访问）
↓
shared（重排）
↓
coalesced 写回
```

---

# 十二、一个经典优化例子（矩阵乘）

---

## ❌ naive：

```cpp
A[row][k]
B[k][col]
```

👉 B 是列访问 → stride 很大 → ❌

---

## ✅ 优化：

```text
把 B load 到 shared memory
↓
转置
↓
warp 连续访问
```

✔ coalesced

---

# 十三、一个形象比喻

想象：

```text
warp = 32 个快递员
内存 = 仓库
```

---

### 好的 coalescing：

```text
32 人去同一货架 → 一次搬完
```

---

### 差的：

```text
32 人去 32 个不同仓库 → 跑 32 次
```

---

# 十四、面试级总结

> **global memory coalescing 是指 warp 内线程的内存访问被合并为尽可能少的内存事务，其效率取决于访问的连续性和对齐性，是 GPU 带宽利用的核心因素。**

---

# 十五、如果你想更深入（强烈建议）

你现在这条线已经到了“性能优化核心区”了，可以继续深入：

我可以带你看：

1. **coalescing 在 SASS 层的实际 transaction 拆分**
2. **L1 / L2 cache 如何影响 coalescing**
3. **Tensor Core（ldmatrix）如何强制 coalesced**
4. **真实 kernel（GEMM / embedding）中的访问优化**

直接说你想看哪一块，我可以带你到“接近 cuBLAS / cutlass 的实现思路”。
