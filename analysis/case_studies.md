# 实例分析：ADOPD 为什么有效

本文用真实生成轨迹说明 ADOPD 训练改变了模型的哪些行为、没有改变什么。所有数字都从保存的生成结果计算，摘录原文截取、未做改写。

## 0. 设置与选样

- **模型对**：4B base vs 4B ADOPD（8k run，step 56，budget 8192）；8B base vs 8B ADOPD（4k run，step 56，budget 4096）。
- **题目**：AMC23、AIME25、GPQA 普通物理各 3 题，每题 4 个样本（temperature 1.0），与 eval 相同的 budget prompt。
- **两种生成方式**：
  - 自由生成：最长 32k。
  - 强制 `</think>`：同一条 rollout 若在 budget 内没结束思考，就截断、插入 `</think>`、只续写答案（总长 ≤ 16k），与训练时一致。
- **两类例子**：
  - §2 **精选示例**：从全部样本中挑出最能说明机制、对我们最有利的样本对，供论文正文使用。
  - §3 **代表性示例**：按固定规则取中位数样本，证明精选现象不是孤例。
  - §4 **局限**：如实列出。
- **启发式指标**：
  - 反思词：Wait / double-check / verify / Let me check / re-examine / Alternatively / Hmm。
  - 自我否定："I made a mistake"、"contradict"、"problem with my assumption" 等。
  - 答案第一次出现的位置：标准答案在推理中第一次出现的位置，只统计数学题。
- **规模**：样本很少（9 题 × 4 样本），用于说明机制；准确率和长度的统计结论以 `todo.md` 的 eval 矩阵为准。完整轨迹在 `tmp/case_study/*.jsonl`。

## 1. 主要发现

### 1.1 总体统计（每个模型 36 个样本）

| 模型 | 自由生成：推理长度 | 推理中的反思词 / 1k token | 强制截断后：答案长度 | 答案中的反思词 | 答案中的自我否定 | 没给出答案 | 给出多个答案 | 强制截断后准确率 | 自己在 budget 内结束 |
|---|---|---|---|---|---|---|---|---|---|
| 4B base | 11,566 | 7.17 | 2,779 | 10.5 | 1.60 | 0.10 | 0.05 | 0.30（n=20） | 0.44 |
| 4B ADOPD（8k，step 56） | 11,092 | 7.23 | 1,560 | 3.1 | 0.75 | 0.05 | 0.00 | 0.30（n=20） | 0.44 |
| 8B base | 14,708 | 7.01 | 4,916 | 30.3 | 2.28 | 0.22 | 0.16 | 0.38（n=32） | 0.11 |
| 8B ADOPD（4k，step 56） | 13,990 | 6.94 | 3,627 | 16.8 | 0.93 | 0.11 | 0.07 | 0.39（n=28） | 0.22 |

"强制截断后"各列只统计被强制插入 `</think>` 的样本；"没给出答案"指答案里没有 `\boxed{}`，通常是写到长度上限仍在推导。

### 1.2 省下的长度来自哪里：答案出现之前 vs 之后

只看 base 和 ADOPD **都答对**的数学题，把自由生成的推理按"标准答案第一次出现"的位置切成两段：

| 模型对 | 题目 | 答案出现前：base → ADOPD | 答案出现后：base → ADOPD | 推理总长：base → ADOPD |
|---|---|---|---|---|
| 4B | aime25 #25 | 10,574 → 11,674 | 3,495 → 3,264 | 14,069 → 14,938 |
| 4B | aime25 #4 | 6,529 → 8,640 | 5,181 → 5,500 | 11,710 → 14,140 |
| 4B | amc23 #19 | 770 → 782 | 2,602 → 1,830 | 3,372 → 2,612 |
| 4B | amc23 #3 | 1,352 → 1,348 | 3,158 → 1,733 | 4,509 → 3,081 |
| 4B | amc23 #40 | 8,218 → 10,486 | 4,696 → 2,415 | 12,914 → 12,901 |
| **4B 平均** | 5 题 | 5,489 → 6,586（+20%） | **3,826 → 2,949（-23%）** | 9,315 → 9,534（+2%） |
| 8B | aime25 #25 | 12,940 → 14,781 | 3,536 → 2,327 | 16,476 → 17,108 |
| 8B | aime25 #4 | 9,003 → 9,076 | 5,481 → 4,994 | 14,484 → 14,070 |
| 8B | amc23 #19 | 970 → 872 | 3,015 → 2,551 | 3,985 → 3,423 |
| 8B | amc23 #3 | 1,365 → 1,415 | 3,066 → 2,588 | 4,431 → 4,003 |
| 8B | amc23 #40 | 10,723 → 12,899 | 12,512 → 4,054 | 23,235 → 16,953 |
| **8B 平均** | 5 题 | 7,000 → 7,808（+12%） | **5,522 → 3,303（-40%）** | 12,522 → 11,111（-11%） |

### 1.3 发现

**发现 1：缩短的不是解题路径，而是"得出答案之后"的冗余验证。**
- 在双方都答对的题上，答案出现之前的推理没有变短，甚至略长：4B +20%，8B +12%。
- 答案出现之后的部分明显变短：4B -23%，8B -40%。
- base 约 40–55% 的推理发生在答案已经得出之后（上表：4B 41%，8B 44%）。最典型的是 AMC23 #3 和 #19：两个模型在几乎同一位置得出答案，之后 ADOPD 早早收尾。
- 这和长度项的设计一致：它不惩罚必要的推导，惩罚的是得出答案后的反复复核。

**发现 2：被强制截断后，接着当前进度作答，而不是推翻重来。**
- 强制截断后，答案中的自我否定（"I made a mistake"、"problem with my assumption"）明显减少：4B 1.60 → 0.75，8B 2.28 → 0.93。
- 答案长度：4B 2,779 → 1,560，8B 4,916 → 3,627。
- 强制截断后的准确率不降：4B 0.30 → 0.30，8B 0.38 → 0.39。
- 这正是 answer-only KL 的作用：teacher 是同一模型在**完整推理之后**写出的答案，学生学到的是"在已有推理上作答"，而不是"在答案区重新推理"。

**发现 3（出乎预期）：base 会在答案区"失控"，ADOPD 很少如此。**
- 被强制截断后，base 经常在答案区一直推导到长度上限都没给出答案：8B 0.22，4B 0.10；ADOPD 降到 8B 0.11，4B 0.05。
- 在多个候选答案之间摇摆（给出多个不同的 `\boxed{}`）：8B 0.16 → 0.07。
- 也就是说，强制截断对 base 不仅"截掉推理"，还可能让它根本交不出答案；ADOPD 学会了在任何截断点都能收尾。

**发现 4：更常在 budget 内自己结束思考。**
- 8B 不需要强制截断的样本比例 0.11 → 0.22。
- 4B 不变（0.44）：8k budget 对 4B 的简单题本来就够用。

**发现 5（出乎预期）：推理"怎么想"没有变，变的是"何时停"和"如何收尾"。**
- 推理中的反思词密度几乎不变（约 7 个 / 1k token）；自由生成时的答案长度也不变（4B 923 vs 919，8B 980 vs 984 token）。
- 训练没有让模型少说 "Wait"，也没有压缩正常作答。
- 它改变的是两个时刻：①得出答案后何时停止验证（发现 1）；②被截断时如何交出答案（发现 2、3）。
- 这也解释了为什么在 32k 自由生成下准确率基本不变：中间的推理过程几乎没动。

## 2. 精选示例（论文正文候选）

### 示例 A：同一解法，base 怀疑自己正确的答案（8B，AIME25，自由生成）

```text
There are $8!= 40320$ eight-digit positive integers that use each of the digits $1, 2, 3, 4, 5, 6, 7, 8$ exactly once. Let $N$ be the number of these integers that are divisible by $22$. Find the difference between $N$ and $2025$.$ Please reason step by step, and put your final answer within \boxed{}.
```

标准答案：`279`

- **base** — 样本 #2：推理 16,387 token，答案 916 token（反思词 0，自我否定 0），最终答案 279，答对。在第 8,095 token 处已经得到 279，又用了 **8,292 token** 复核。
- **ADOPD** — 样本 #3：推理 9,748 token，答案 1,252 token（反思词 0，自我否定 0），最终答案 279，答对。在第 9,666 token 处得到 279，之后只用了 **82 token** 就收尾。

**base** 得到答案的那一刻，以及之后：

```text
e, total N=16 * 4! * 3!.

Wait, this is different from what I thought before. Earlier, I thought that for each d, N_d=4*... but actually, the total is (sum of C_d) * 4! * 3!.

Since sum of C_d is 4 + 4 + 4 + 4 = 16. So N=16 * 24 * 6 = 16 * 144 = 2304. So N=2304.

Therefore, the answer N is 2304. Then, the problem says to find N - 2025. So 2304 - 2025 = ?

Compute 2304 - 2025:

2304 - 2000 = 304

304 - 25 = 279.

Therefore, the difference is 279. Therefore, the answer is 279. But let me check my reasoning again because 2304 seems high? Let me verify.

Alternative approach: Let me think about the total number of permutations where the number is divisible by 22. Since divisibility by 22 requires divisibility by 2 and 11. Since divisibility by 2 is already considered by fixing last digit even (4 options). Divisibility by 11 requires that the sum of odd positions equals sum of even positions. Since we have shown that this requires sum of each to be 18.

Now, the number of such permutations can be calculated as follows:

First, choose the l
```

……之后它尝试了生成函数等其他思路，其中一条是错误的，绕了一圈才回到原答案。推理的结尾：

```text
ncile this with the generating function approach?

The earlier calculation with the generating function was incorrect because it assumed that all subsets S (of size 4) are valid (i.e., sum to 18), which is not true. The generating function approach I did earlier was for all subsets S (of size 4), not just those that sum to 18. Therefore, that calculation was incorrect.

Therefore, the correct number_of_pairs is 16, and hence N=16 * 4! * 3! = 2304, which is consistent with our initial calculation.

Therefore, I can be confident that N=2304, and hence the answer is 2304 - 2025 = 279.

Thus, the final answer is 279.

**Final Answer**
\boxed{279}
```

**ADOPD** 在算最终差值之前，先用另一种方法核对中间量 N = 2304，然后直接收尾：

```text
re position 8 is even is (number of even digits in subset) * 3! * 4! Therefore, total is 8 subsets * [k * 3! * 4!], where k=2 for each subset, so 8*2*6*24= 2304. Alternatively, the total number of permutations with Sum_even=18 is 8*4!*4! = 4608. The fraction of those where position 8 is even is equal to the fraction of permutations where, given the subset, position 8 is even. For each subset, the probability that position 8 is even is (number of even digits in subset)/4. Therefore, total number is 4608 * average(k/4). Since each subset has k=2, average(k/4)= 2/4=1/2. Therefore, total number is 4608 * 1/2 = 2304. Which matches. Therefore, this confirms that N=2304.

Therefore, the answer N is 2304. Then, the problem says "Find the difference between N and 2025." So compute 2304 - 2025.

Let me compute that: 2304 - 2000 = 304; 304 - 25 = 279. Therefore, 2304 - 2025 = 279.

Wait, let me check:

2025 + 279 = 2304?

2025 + 200 = 2225; 2225 + 79 = 2304. Yes. Therefore, the difference is 279.

Therefore, the answer is 279. Therefore, boxed answer is \boxed{279}

**Final Answer**
\boxed{279}
```

两个模型用的是同一个解法、得到同一个 N = 2304：
- base 在得出 279 后又怀疑 "2304 seems high?"，花了一半推理重新推导；
- ADOPD 把核对放在中间量上，得出答案后就收尾，总推理短了约 40%。

### 示例 B：被截断后，接着算完，而不是推翻设定（8B，GPQA 物理，budget 4096）

```text
An electron is moving along the horizontal axis from the left with a Lorentz factor of 4. A positron is also moving along the horizontal axis, but from the right with a Lorentz factor of 2. After colliding, both annihilate producing 2 photons.

The energies of both photons were measured and they have the same value. The directions were also measured. One of the photons was moving in the upper-right direction, while another in the lower-right direction.

What is the angle between the directions of the photons?
A. 96
B. 74
C. 172
D. 138

Please reason step by step, and put your final answer (only the choice letter A, B, C, or D) within \boxed{}.
```

标准答案：`D`

4 个样本：base 4/4 被强制截断，答案 906–8,124 token；ADOPD 2/4 被截断，另外 2 个在 budget 内自己结束；两者全部答对。

- **base** — 样本 #1：推理 4,096 token，答案 6,990 token（反思词 47，自我否定 21），最终答案 D，答对，在 budget 处被强制插入 `</think>`。`</think>` 之后：

```text
Now, since sqrt(3) - sqrt(15) is approximately -2.14, this gives cos(theta) ≈ -2.14 / 6 ≈ -0.3567. Therefore, theta is approximately 110.7 degrees. But this suggests that each photon makes an angle of about 110.7 degrees with the horizontal. However, this would imply that the photons are pointing mostly to the left, which contradicts the problem's statement that they are moving in upper-right and lower-right directions. 

Clearly, there's a problem with my assumption in setting up the angle theta. Perhaps I need to consider the direction of the photons more carefully. 

Wait, the problem states that one photon is moving upper-right and the other is lower-right. If the initial momentum is to the left (negative x), then the final momentum must be to the left. Therefore, the sum of the photons' momenta must be to the left. 

However, if the photons are moving upper-right and lower-right, their horizontal components are both to the right (since they're in the right direction). Wait, but that would make the net horizontal momentum to the right, which contradicts the initial momentum to the left. Therefore, there must be a mistake in my earlier assumption. 

This suggests that my setup of the final momentum is incorrect. Let me re-examine the directions of the photons. 

If the photons are moving upper-right and lower-right, their horizontal components are to the right. However, thei

[... 省略 24,391 个字符 ...]


```

- **ADOPD（被截断的样本）** — 样本 #2：推理 4,096 token，答案 2,194 token（反思词 8，自我否定 0），最终答案 D，答对，在 budget 处被强制插入 `</think>`。`</think>` 之后：

```text
Now, compute the numerator:

5.605 * (3.87298 + 1.73205) ≈ 5.605 * 5.605 ≈ 31.416

Denominator is 12. So tan(theta) ≈ 31.416 / 12 ≈ 2.618

Ah! Wait, that's interesting. 31.416 / 12 is exactly 2.618. Which is approximately the golden ratio (≈1.618), but 2.618 is actually 1.618 * 1.618 ≈ 2.618. 

So tan(theta) ≈ 2.618. 

Now, let me find arctangent of 2.618. 

I know that tan(69.3°) ≈ 2.618. Let me check:

Using calculator input: tan(69.3°) ≈ tan(69 + 0.3 degrees). 

Alternatively, using exact value: 2.618 is approximately equal to (sqrt(5)+3)/2 ≈ (2.236 + 3)/2 ≈ 5.236/2 ≈ 2.618. 

So, tan(theta) = (sqrt(5)+3)/2. 

Let me check if this is true. Let me compute (sqrt(5) + 3)/2 ≈ (2.236 + 3)/2 ≈ 5.236/2 ≈ 2.618, yes. 

So, tan(theta) = (sqrt(5)+3)/2. 

Now, what angle has tangent equal to (sqrt(5)+3)/2? 

Let me think. Let me recall that tan(75°) = 2 + sqrt(3) ≈ 3.732. tan(60°) = sqrt(3) ≈ 1.732. tan(65°) ≈ 2.144. tan(67.5°) = tan(π/2.285...) ≈ 2.414. Wait, tan(67.5°) is tan(45° + 22.5°) = (tan45 + tan22.5)/(1 - tan45 tan22.5) = (1 + 0.4142)/(1 - 0.4142) ≈ 1.4142 / 0.5858 ≈ 2.4142. 

So, tan(theta) ≈ 2.618 is between 67.5° and 75°, closer to 69.3°, as earlier. 

But perhaps there is an exact value? Let me think. 

Alternatively, maybe theta is 69.3 degrees, so angle between photons is

[... 省略 4,601 个字符 ...]


```

- **ADOPD（自己结束的样本）** — 样本 #3：推理 3,095 token，答案 1,247 token（反思词 0，自我否定 0），最终答案 D，答对，在 budget 内自己结束思考。思考的最后一段：

```text
he calculation seems correct.

Another check: Let me compute the exact value of cosθ.

sqrt(15) is approx 3.87298, sqrt(3) is approx 1.73205. So sqrt(15)-sqrt(3) ≈ 2.14093. Divided by 6 is approx 0.35682. So arccos(0.35682). Let me compute it more accurately.

Using calculator input: arccos(0.3568). Let me recall that cos(60°) is 0.5, cos(69°) ≈ 0.3584, which is very close to 0.3568. So θ ≈ 69.0 degrees. So 2θ ≈ 138 degrees. Therefore, answer D.

Alternatively, maybe there's a way to compute it exactly? Let me see. Let me compute sqrt(15) - sqrt(3):

But perhaps there's exact value? Not sure. But the approximate calculation gives us 138 degrees, which is option D. So I think the answer is D.
```

截断发生在同一处（4096 token），两者的做法截然不同：
- base 在答案区发现动量方向"矛盾"，推翻自己的设定，从头推导，写了 7k token；
- ADOPD 接着截断时算到一半的式子，得到 tan θ ≈ 2.618 → θ ≈ 69.3° → 夹角约 138°，选 D。

### 示例 C：base 写到长度上限都没交出答案，ADOPD 简洁答对（8B，GPQA 物理，budget 4096）

```text
Consider an isolated system of 13 identical particles each with charge 2e and negligible mass. 12 of these charges are
constrained to stay at 2 m from a point P. The 13th charge is fixed at P. What is the minimum energy (in Joules) of this system
correct to three decimals?
A. 122.330
B. 2.822 x 10^-26
C. 5.645 x 10^-26
D. 7.056 x 10^-27

Please reason step by step, and put your final answer (only the choice letter A, B, C, or D) within \boxed{}.
```

标准答案：`B`

- **base** — 样本 #0：推理 4,096 token，答案 12,287 token（反思词 85，自我否定 10），最终答案 无，答错，在 budget 处被强制插入 `</think>`，总长 16,384 token，到达 16k 上限。答案区里的自我否定句（共 7 句，节选）：

```text
But perhaps I made a mistake in the arrangement of the charges.
But maybe I made a mistake in calculation of the number of terms or in the formula.
Wait, perhaps I made a mistake in the values of the constants.
Alternatively, maybe I made a mistake in the sign of the potential energy? All charges are positive, so the potential energy is positive, which matches the calculation.
Alternatively, perhaps I made a mistake in the calculation of the total energy.
Alternatively, maybe I made a mistake in the formula for the potential energy.
```

base 答案区的最后一段（仍在推导，没有结论）：

```text
ergy between the 13th charge and itself? No, it's a single charge. 

Wait, but all charges are identical? Yes, all 13 particles have charge 2e. 

Wait, maybe the problem is asking for the minimum potential energy, but I need to consider that when charges are brought together, the potential energy is lower? No, since all charges are positive, bringing them closer increases the potential energy. 

Wait, but the minimum potential energy is achieved when the charges are as far apart as possible, which is what I assumed. 

Alternatively, perhaps the minimum energy is achieved when the 12 charges are all at the same point? No, but they are constrained to stay at 2 m from P, so they can't be at the
```

- **ADOPD** — 样本 #1：推理 4,096 token，答案 1,423 token（反思词 0，自我否定 0），最终答案 B，答对，在 budget 处被强制插入 `</think>`。答案区的开头：

```text
To determine the minimum energy of the system, we analyze the electric potential energy contributions from all pairs of charges. The system consists of 13 identical particles, each with charge $ q = 2e $, where $ e = 1.602 \times 10^{-19} \, \text{C} $. The 13th charge is fixed at point $ P $, while the other 12 charges are constrained to stay 2 meters from $ P $.

---

### **Step 1: Potential Energy Between the Central Charge and the Other 12 Charges**

Each of the 12 charges is at a distance of 2 m from the central charge. The potential energy between a pair of charges is given by:
$$
U = k \frac{q_1 q_2}{r}
$$
Here, $ k = 8.988 \times 10^9 \, \text{N·m}^2/\text{C}^2 $, $ q_1 = q_2 = 2e $, and $ r = 2 \, \text{m} $. Substituting:

$$
U = k \cdot \frac{(2e)(2e)}{2} = k \cdot \frac{4e^2}{2} = 2k e^2
$$

Since there are 12 such charge pairs, the total potential energy from this part is:

$$
U_{\text{central}} = 12 \cdot 2k e^2 = 24k e^2
$$

---

### **Step 2: Potential Energy Between the 12 Charges at 2 m from P**

This is a complex term because the 12 charges are constrained to remain 2 m from $ P $, but their relative positions are not fixed. The configuration that minimizes the potential energy of the system (since all charges are positive and repel each other) is a regular 12-gon (dodecagon), which maximizes the separation between the charges.

To compute the total potential energy, we consider all pairs of the 12 charges and sum their contributions. In a regular polygon o

[... 省略 2,234 个字符 ...]


```

4B 上同一道题也有完全相同的现象：base 样本 #2：推理 8,192 token，答案 8,191 token（反思词 64，自我否定 18），最终答案 无，答错，在 budget 处被强制插入 `</think>`；ADOPD 样本 #1：推理 8,192 token，答案 1,060 token（反思词 0，自我否定 0），最终答案 B，答对，在 budget 处被强制插入 `</think>`。

### 示例 D：同一时刻得到答案，ADOPD 不再换方法反复验证（4B，AMC23，自由生成）

```text
What is the value of
\[2^3 - 1^3 + 4^3 - 3^3 + 6^3 - 5^3 + \dots + 18^3 - 17^3?\] Please reason step by step, and put your final answer within \boxed{}.
```

标准答案：`3159`

- **base** — 样本 #3：推理 4,945 token，答案 765 token（反思词 0，自我否定 0），最终答案 3159，答对；第 1,324 token 得到 3159，之后又用了 3,621 token。
- **ADOPD** — 样本 #3：推理 3,217 token，答案 640 token（反思词 0，自我否定 0），最终答案 3159，答对；第 1,297 token 得到 3159，之后用了 1,920 token。

base 推理的结尾（得到答案后又换第二、第三种方法确认）：

```text
 3³ + 5³ + ... + 17³. Which we calculated as 13041.

So difference is 16200 - 13041 = 3159. 

Alternatively, maybe I can check the sum of odd cubes. Let me compute that manually. Let me compute 1³ + 3³ + 5³ + ... + 17³.

Compute each term:

1³ = 1

3³ = 27

5³ = 125

7³ = 343

9³ = 729

11³ = 1331

13³ = 2197

15³ = 3375

17³ = 4913

Now, let me add these up step by step:

Start with 1 + 27 = 28

28 + 125 = 153

153 + 343 = 496

496 + 729 = 1225

1225 + 1331 = 2556

2556 + 2197 = 4753

4753 + 3375 = 8128

8128 + 4913 = 13041. 

Yes! That's exactly the same as before. So that's correct. Therefore, the difference is indeed 16200 - 13041 = 3159. 

Therefore, I can be sure that the answer is 3159. 

Alternatively, maybe there's another formula or a way to think about this. For example, considering that each pair is (even)^3 - (odd)^3, and since even numbers are 2k and odd numbers are 2k - 1, then the difference can be expressed as (2k)^3 - (2k - 1)^3 = 12k² - 6k + 1, as we found earlier. Then summing that over k from 1 to 9 gives 3159. 

Alternatively, if I think about the entire sum as the sum of all even cubes minus the sum of all odd cubes up to 17³. Which is the same as we did before.

All methods lead to 3159, so I think that is the correct answer.

**Final Answer**
\boxed{3159}
```

ADOPD 推理的结尾（只做一次逐对求和核对）：

```text
Let me try:

First pair: 8 -1=7

Second pair: 64 -27=37. Total so far: 7+37=44

Third pair: 216 -125=91. Total: 44+91=135

Fourth pair: 512 - 343= 169. Total: 135 +169= 304

Fifth pair: 1000 - 729=271. Total: 304 +271=575

Sixth pair: 1728 - 1331= 397. Total: 575 + 397= 972

Seventh pair: 21952? Wait, wait, 14³ is 2744, 13³ is 2197. Wait, no, seventh pair is 14³ -13³? Wait, no. Wait, the pairs are (2k)^3 - (2k-1)^3. For k=7: 14³ -13³. 14³ is 2744, 13³ is 2197. 2744 - 2197= 547. So adding that to 972: 972 + 547= 1519.

Eighth pair: 16³ -15³. 16³ is 4096, 15³ is 3375. 4096 -3375=721. Total: 1519 +721=2240.

Ninth pair: 18³ -17³. 18³ is 5832, 17³ is 4913. 5832 -4913= 919. Total: 2240 +919= 3159. Yes! So adding all pairs step by step gives the same result. Therefore, the answer is 3159. So I think that's correct.

Therefore, the value of the expression is 3159.

**Final Answer**
\boxed{3159}
```

## 3. 代表性示例（中位数样本）

选样规则：在答对的样本中取长度中位数（自由生成看推理长度，强制截断看答案长度），全错时在全部样本中取。每行是同一道题上 base vs ADOPD 的 4 样本平均值。

| 题目 | 方式 | 准确率：base → ADOPD | 推理：base → ADOPD | 答案：base → ADOPD | 现象 |
|---|---|---|---|---|---|
| 4B amc23 #3 | 自由 | 1.00 → 1.00 | 4,509 → 3,081 | 833 → 610 | 得出答案后少了复核 |
| 8B gpqa_physics_general #reciuppd | 强制 | 1.00 → 1.00 | 4,096 → 3,734 | 4,315 → 2,040 | 截断后接着算完，答案变短 |
| 4B aime25 #12 | 强制 | 0.00 → 0.00 | 8,192 → 8,192 | 5,378 → 1,132 | 推理不够时直接给出简洁的解（都答错） |
| 8B amc23 #19 | 强制 | 1.00 → 1.00 | 3,852 → 3,361 | 858 → 703 | 更多样本自己结束思考 |
| 8B aime25 #12 | 自由 | 0.00 → 0.00 | 31,708 → 31,192 | 466 → 642 | 局限：难题推理长度不变 |

## 4. 局限

- **难题的推理长度不变。** 8B 在 AIME25 #12 上自由生成：base 推理 31,708 token，ADOPD 31,192 token，都答错，ADOPD 的反思词还略多。全量 eval 中 AIME/HMMT 的长度只缩短 3–6%，缩短集中在 AMC23 和 GPQA。长度项只在能完成的题上起作用。

- **budget 不够时，"直接作答"意味着补不回缺失的推理。** 4B AIME25 #12 在强制截断时：base 答案 5,378 token，ADOPD 1,132 token，都答错。base 在答案区继续推理，偶尔能靠这一点"救回来"；ADOPD 放弃了这条路。这与全量 eval 中紧 budget 下 forced 准确率的下降一致（4B 2k：0.528 → 0.477）。

- **个别题目准确率下降。** 4B GPQA 物理（rec7qmSn）自由生成准确率 0.75 → 0.25，推理长度几乎相同。每题只有 4 个样本，可能是噪声，但不能排除训练带来的退化。

- **答案出现之前的推理略长**（4B +20%，8B +12%）。方法不会缩短解题路径本身；要压缩这部分，需要别的机制。

- **样本与指标的限制。** 只有 9 题 × 4 样本。反思词、自我否定和答案位置都是基于字符串的启发式指标，"答案第一次出现"只适用于数值答案的数学题。

## 5. 小结（可用于论文）

1. **ADOPD 不改变模型"怎么推理"，改变的是"何时停"和"如何收尾"。** 推理中的反思密度不变；双方都答对的题上，答案出现之前的推理不变，答案出现之后的冗余验证减少 23–40%。
2. **answer-only KL 让模型在任何截断点都能交出答案。** teacher 是同一模型在完整推理之后的作答，学生据此学会了"在已有推理上直接作答"。被强制截断后，自我否定减少约一半，没交出答案的比例减半，准确率不降。
3. **局限**：难题的推理长度基本不变；budget 太紧时，直接作答意味着放弃在答案区补推理的机会，准确率随之下降。
