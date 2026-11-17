---
layout: "blog-post"
title: "Staleness 在左，Throughput 在右：异步 RL 框架漫谈（二）"
description: "我们在 RL 模拟器的统一环境中比较各框架的性能，分析有效吞吐、丢弃浪费、长尾等待与数据选择偏向。"
date: "2026-11-14"
lang: "zh-CN"
series: "PSRL / 02"
permalink: "/blog/async-rl-staleness-2/"
redirect_from: ["/blog/simrl-staleness/"]
list_order: 2
cover: "/assets/blog/async-rl-staleness-2/cover.svg?v=427901ed109d"
cover_alt: "14 个框架配置在 staleness 上限 1、2、3 下的端到端有效吞吐对比"
cover_width: 3200
cover_height: 1920
author_profile: false
---

# Staleness 在左，Throughput 在右：异步 RL 框架漫谈（二）

在上一篇博客中，我们分析了不同 RL 框架的 staleness 控制机制。在结尾，我们留下了一个很现实的问题：

**这些控制机制在实际运行中的表现究竟如何？**

例如，只在 L1 限制准入量，究竟能把 staleness 控制到什么程度？在训练前丢弃超限数据，虽然可以守住上限，但已经完成的生成计算会浪费多少？如果不想丢弃数据，是否就要让训练侧等待更久？在这篇博客中，我们将尝试解答这些问题。

> **源码说明**：源码核对日期为 2026-10-16，选用当时各仓库默认分支的最新 commit，与第一篇博客一致。本文复现的机制以那次核对时的逻辑为准。

## Part I：实验方案

### 1. 在同一环境中比较控制机制


要比较这些机制，最直接的办法是把各个框架都部署到真实集群上，再分别调整参数、记录结果。但这样做不仅部署和实验成本高，还会遇到一个更关键的问题：如何保证不同运行之间的可比性？不同框架所用的推理引擎、训练后端和依赖版本各不相同。即使硬件相同，每次运行采到的 response 长度、长尾出现的位置也会变化，而 staleness 恰恰对这些因素十分敏感。这样一来，就很难判断观测到的差异究竟来自控制机制，还是来自执行后端和负载本身。

为此，我们搭建了一套 RL 模拟器 [**SimRL**](https://github.com/psrl-project/SimRL)，在相同的硬件、模型和负载设置下复现第一篇博客中讨论的各类控制机制。SimRL 模拟 RL 后训练系统的整套运行逻辑：rollout 侧采用 continuous batching 并进行 KV cache 管理，训练侧通过拟合的成本模型估计每一步耗时，而 L1–L3 的准入、取消和过滤或等待等控制机制则可分别插装配置。这样，我们就能排除其他可能带来干扰的因素，专注于观察 RL 框架的机制本身所产生的影响。

因此，后文比较的是各套 staleness 控制逻辑在 SimRL 统一模拟环境中的表现。各框架自身的推理引擎、kernel 和工程开销等因素并未纳入考量，所以后文的吞吐不能直接代表它们在真实集群上的性能。

SimRL 本身的设计将在后续博客中详细介绍，这里我们只用一个简单的例子来说明 SimRL 模拟的可靠性。


### 2. 模拟与真实运行的对照

我们用一次真实的 verl `drop` 运行来检查模拟器的表现（先运行 verl，记录其真实输入输出长度的 trace，再使用模拟器运行该 trace，训练时长采用真实运行的测量值）。下图比较了真实运行与模拟运行中 rollout 侧请求总数随时间的变化，以及 KV cache 占用（两个 rollout instance 取最大，对应右侧纵轴）。背景阴影标出训练阶段，其中斜纹部分是 PPO 所需的 log-prob 重算（`old_log_prob`），实色部分是梯度更新（`update_actor`）。

[![真实 verl 运行与 SimRL 模拟对比]({{ '/assets/blog/async-rl-staleness-2/verl_alignment-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/verl_alignment-blog.png' | relative_url }})

这组对照采用如下配置：最大 staleness 以及`max_warmup_batches` 均设置为1，并采用 `drop` 的策略。每个 batch 包含 $$32$$ 个 group，每个 group 一共 $$8$$ 条 trajectory。从统计量看，模拟相对真实运行的总时长偏差为 $$-1.5\%$$，忙时平均在途请求数偏差为 $$+1.5\%$$，忙时平均 KV 占用偏差为 $$-1.6\%$$。这基本印证了通过 SimRL 的模拟，我们可以较为准确地反映真实系统的运行情况。

### 3. 实验设置

后文所有模拟实验均采用以下模型、资源和算法设置：

- 模型为 Qwen3-8B，一共 $$8$$ 张 H20。
- rollout 用 $$2$$ 个 instance，每个 TP1，占 $$2$$ 张卡。
- 训练用 Megatron，TP2 / PP3 / DP1，占 $$6$$ 张卡。
- GRPO，group size 为 $$8$$，每个训练步取 $$16$$ 个 prompt group。
- 默认每次运行训练 $$10$$ 个版本，共计 $$1280$$ 条 trajectory。
- 负载是真实的 dapo-math-17k 的 trace，response 上限截断在 $$38912$$ token。
- 每个 rollout instance 的 KV 池为 $$31262$$ 个 block，即 $$500192$$ token，这是在同机 H20 上按同一份 checkpoint 实测出来的容量。KV 池容量会影响一个 instance 能容纳多少条在途 trajectory，以及何时开始抢占。后文涉及准入量和吞吐的分析都与这一容量有关。更换硬件或 KV 池设置后，需要重新测量。

负载的长度分布如下：

[![dapo-math-17k 负载的 prompt 与 response 长度分布]({{ '/assets/blog/async-rl-staleness-2/workload_lengths-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/workload_lengths-blog.png' | relative_url }})

### 4. 衡量指标

为了衡量系统真正提供给 RL 训练的数据量，我们采用**有效吞吐**作为性能指标：

$$
\begin{equation*}
\text{有效吞吐}=\frac{\text{进入训练 batch 的 token 数}}{\text{这次运行的总时长}}
\end{equation*}
$$

生成后被丢弃的 token 不计入分子。而相应地，**全量生成吞吐**统计所有生成过的 token，包括最终被丢弃的部分，分母仍为同一次运行的总时长。我们用 **waste** 表示被丢弃 token 占全部生成 token 的比例，因此有：

$$
\begin{equation*}
\text{有效吞吐}=\text{全量生成吞吐}\times(1-\text{waste})
\end{equation*}
$$

一个配置即使生成得很快，也可能因为大量丢弃而只能提供较低的有效吞吐。后文的柱状图会同时展示这两部分：

- **实色部分表示有效吞吐**，按实际参与训练的 token 的 staleness 由浅到深分层。
- **斜线阴影表示被丢弃的部分**，与实色部分相加得到全量生成吞吐。涉及 L2 取消和 L3 过滤时，两种阴影分别标出它们造成的丢弃。

此外，除了比较不同配置下的吞吐，我们还关注不同配置下实际参与训练的 token 的**真实最大 staleness** 是否会超过该参数配置下所预期的最大 staleness，来判断框架是否具备严格控制 staleness 的能力。

## Part II：各框架的实验结果

在进入实验之前，我们先回顾一下第一篇博客中各框架的控制位置和关键参数。下图最后一列给出了第一篇博客中基于源码进行理论分析后得出的结论，而本文则希望通过实验进一步校验这些机制的实际表现。

[![各框架的控制位置、关键参数与严格 staleness 控制（理论分析）]({{ '/assets/blog/async-rl-staleness-2/control-mechanisms-recall-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/control-mechanisms-recall-blog.png' | relative_url }})

上表中，对于每个框架，我们用蓝色字体标记出了实验中核心调节的参数，其通常是整个框架中与 staleness 上限最直接相关的旋钮，也是用户在考虑 staleness 控制时会最优先使用到的参数。下文将这类参数统一记为 $$\eta$$，它表示对应配置的取值，是否等于实际 staleness 上限还要看具体机制。其余次要参数（会对 staleness 造成影响，但并不是主要控制旋钮）我们大多维持在框架的默认值（个别情况会在后文具体说明）。下面展示各框架的实验结果。

<details markdown="1" class="framework-detail" id="framework-verl">
<summary>verl</summary>

verl 有两个核心参数：`num_warmup_batches` 决定[启动时额外预取](https://github.com/verl-project/verl/blob/5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517/verl/trainer/ppo/v1/trainer_base.py#L624-L651)多少批 prompt，对应 L1 准入。`max_off_policy_threshold` 决定 L3 的超限阈值，再由 `max_off_policy_strategy` 选择 [丢弃 `drop`](https://github.com/verl-project/verl/blob/5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517/verl/trainer/ppo/v1/replay_buffer.py#L520-L539) 或 [等待 `wait`](https://github.com/verl-project/verl/blob/5b550b0c3dad8c7bb7aab834a3fe195f8cdfc517/verl/trainer/ppo/v1/replay_buffer.py#L541-L556)。本文分别将策略固定为默认的 `drop` 和作为对照的 `wait`，每个子图固定一个 `num_warmup_batches`，横轴扫描阈值 `max_off_policy_threshold`。阈值为 $$\eta$$ 时，预期的 token staleness 上限为 $$\eta-1$$。

**(1) `drop` 的实验结果**

[![verl drop：准入与上限是两个参数]({{ '/assets/blog/async-rl-staleness-2/sweep_verl_drop-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_verl_drop-blog.png' | relative_url }})

**`drop` 在全部 $$25$$ 个点上都守住了上限。** 其中，warmup 为 $$1$$、阈值为 $$5$$ 时，实际最大 staleness 为 $$3$$。其余 $$24$$ 个点则恰好达到对应上限。可见，其具备严格控制 staleness 的能力。

当我们固定 warmup 时，增大阈值，可以明显减少 waste 并提高有效吞吐，这非常符合直觉。但有意思的是，固定阈值、增大 warmup，反而会让最终有效吞吐变低。原因在于，warmup 增大后，同一生成版本上会提前准入更多 group，但训练每次仍只消费一个 batch。后面才被取用的结果就更容易超限，在 L3 被丢弃。它们虽然占用了生成资源，却没有成为有效训练数据。例如，阈值固定为 $$1$$ 时，warmup 从 $$1$$ 增至 $$5$$，waste 从 $$61.2\%$$ 升至 $$87.5\%$$。可见，**增加预取量是否有利，要看提前生成的数据最终能否被训练采用**，而不能只看并发是否更高。

**(2) `wait` 的实验结果**

[![verl wait：超限后只等待，不丢弃]({{ '/assets/blog/async-rl-staleness-2/sweep_verl_wait-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_verl_wait-blog.png' | relative_url }})

切换到 `wait` 后，尽管不会出现任何 waste，但其中 $$15$$ 个点的实际最大 staleness 均超过了预期的上限，并不具备严格控制 staleness 的能力。

与 `drop` 的观察类似，固定 warmup 增大阈值时，整体的有效吞吐基本上呈增长趋势，而固定阈值增大 warmup 时则恰恰相反。这导致最终性能最优的点全部出现在 verl 默认的 `num_warmup_batches=1` 的配置下。

此外，在这份负载下，我们可以观察到一个有意思的现象：似乎只要把阈值设得比 warmup 大，就可以避免超限。那么这个规律是否对其他负载也成立呢？其实不然。

我们先从机制上分析：`wait` 只在还有未完成的 group 达到阈值时暂停训练取样，一个 group 完成后就不再受这一检查约束。如果同一生成版本上积压了多批慢组，训练可能先消费已经完成的快组，随后因慢组达到阈值而停下来。等慢组都完成，等待便会解除，但它们不一定能一次取完。训练每次只消费一批，剩余的同版本结果在后续取用时又会增加版本差，而且不会被过滤。因此，即使阈值大于 warmup，也仍可能超过预期的 staleness 上限。

为了验证我们的分析，我们另外构造了一份负载：按输入顺序先安排一批快组，接着两批慢组（每个 group 的每条 response 都取 $$25000$$ 到 $$31000$$ token），后续批次则全部是快组。warmup 为 $$2$$ 或 $$3$$ 时，启动准入会将前面的多批数据放在同一个生成版本上，快组则让训练能够在慢组完成前继续推进。

[![构造负载的 32 个输入批次长度分布，第 2、3 批全部为长 response，下方放大快批次]({{ '/assets/blog/async-rl-staleness-2/slow_batch_lengths-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/slow_batch_lengths-blog.png' | relative_url }})

可以看到，这份负载共有 $$32$$ 批，每批 $$128$$ 条 response。第 $$2$$、$$3$$ 批的所有 response 都较长，其余 response 则为 $$300$$ 到 $$700$$ token。实验结果如下：

[![verl wait：整批慢组，十个点全部超限]({{ '/assets/blog/async-rl-staleness-2/sweep_verl_wait_cohort-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_verl_wait_cohort-blog.png' | relative_url }})

在这份构造负载上，warmup 取 $$2$$ 和 $$3$$，阈值扫描 $$1$$ 到 $$5$$，**十个点全部超限**，其中包括阈值大于 warmup 的五个点。

事实上，使用 `wait` 确实能保证一个 staleness 上界，不过该上界其实远比我们想的要大。

记 $$W$$ 为预热参数 `num_warmup_batches`，$$\eta$$ 为阈值 `max_off_policy_threshold`，且两者均取正整数，那么所有参与训练的 token 都满足：

$$
\begin{equation*}
\text{token staleness} \leq\max(W, \eta+W-2).
\end{equation*}
$$

这个上界与我们直观期待的 $$\eta-1$$ 并不相同，而且还依赖预热量 $$W$$。例如，$$W=2$$、$$\eta=3$$ 时，上界是 $$3$$，而预期限制只有 $$2$$。感兴趣的读者可以展开下面的证明。

<details markdown="1" class="proof-detail" id="verl-wait-bound-proof">
<summary>为什么上界是 \(\max(W,\eta+W-2)\)？</summary>

**先明确证明的条件。** 设一个训练 batch 包含 $$B$$ 个 group，$$B$$ 为正整数，训练版本从 $$0$$ 开始计数。启动时，系统额外预取 $$WB$$ 个，再加上首轮正常准入的 $$B$$ 个，共 $$(W+1)B$$ 个。之后，训练每推进一个版本，只新增 $$B$$ 个准入额度，每次取样也恰好消费 $$B$$ 个 group。取样优先选择**准入版本最老的已完成 group**，不会用更新版本的数据越过已经就绪的旧版本数据。

此外，`wait` 不丢弃数据，但只要还有未完成 group 满足“当前训练版本减准入版本，再加 $$1$$，不小于 $$\eta$$”，就不能继续取样。

固定一个准入版本 $$a$$，考察在这一版本准入的任意 group，同时把准入版本不超过 $$a$$、尚未被消费的 group 称为旧 group。记当前训练版本为 $$v$$。一条 token 的生成版本不早于其 group 的准入版本，因此，我们要证明的就是这个 group 被取用时 $$v-a$$ 的上界。

**首先，计算最多还会积压多少旧 group。** 在版本 $$a$$ 本轮取样之前，累计准入额度最多为 $$(W+1+a)B$$，此前已经消费 $$aB$$ 个 group，所以未消费总量最多为 $$(W+1)B$$。版本 $$a$$ 再消费一批后，准入版本不超过 $$a$$ 的未消费 group 最多剩 $$WB$$ 个。此后新增的 group 都属于更新版本，不会增加这部分旧数据的数量。

**当 $$\eta=1$$ 时，等待条件在 group 准入的当版就生效。** 所以训练不能越过尚未完成的 group。版本 $$a$$ 取样前至多积压 $$(W+1)B$$ 个旧 group，它们就绪后会优先被取走，最多占 $$W+1$$ 个 batch。第一批使用版本 $$a$$，最后一批至迟使用版本 $$a+W$$，故 $$v-a\leq W$$。若 group 在本轮取样之后才准入，则旧数据最多剩 $$WB$$ 个，从下一版开始取用，最后一批同样不晚于 $$a+W$$。

**当 $$\eta\geq2$$ 时，考察版本 $$a+\eta-1$$。** 到达这一版本后，只要仍有准入版本不超过 $$a$$ 的 group 尚未完成，等待检查就会阻止取样。因此，这一轮真正能够取样时，所有剩余旧 group 都已完成。此时已经过版本 $$a$$ 的首次取样，旧数据最多剩 $$WB$$ 个，又因为旧版本优先，最多再用 $$W$$ 个 batch 就能全部取完。最后一批使用的版本至迟为：

$$
\begin{equation*}
\begin{aligned}
v&\leq(a+\eta-1)+(W-1)\\
 &=a+\eta+W-2.
\end{aligned}
\end{equation*}
$$

在该门限之前就已经被消费的 group，版本差小于 $$\eta-1$$，也满足这一界。若运行较早结束，只会使实际能达到的版本差更小。

合并两种情况，即得到：

$$
\begin{equation*}
v - a\leq\max(W,\eta+W-2).
\end{equation*}
$$

此即为真实的 staleness 上限。

</details>

**(3) 对比与讨论**

固定 warmup 为 $$1$$，将两种策略按实际最大 staleness 对齐，相同时取扫描中最高的有效吞吐：

[![warmup 为 1 时，drop 与 wait 在相同实际最大 staleness 下的有效吞吐，叉号表示本次扫描未达到该实际值]({{ '/assets/blog/async-rl-staleness-2/verl_drop_wait_matched-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/verl_drop_wait_matched-blog.png' | relative_url }})

从图中我们可以看到，实际最大 staleness 为 $$1$$、$$2$$ 时，`wait` 的吞吐更高，分别约为 $$1825$$ 和 $$1864$$ tokens/s，`drop` 为 $$1436$$ 和 $$1816$$。为 $$3$$ 时两者持平，都是 $$1945$$。

因此，两种策略各有取舍：`wait` 会保留已经生成的结果，在这组实验中因此取得了更高或相同的有效吞吐。不过，它的阈值不能直接当作实际 staleness 上限，配置时还要考虑具体负载和 warmup 的设置。`drop` 则能直接排除超限结果，让 staleness 的控制更严格，代价是可能浪费已经完成的生成。可见，`wait` 有更高的吞吐，`drop` 则有更明确的版本约束，实际选用哪一种，需结合任务需求具体判断。

</details>

<details markdown="1" class="framework-detail" id="framework-trl">
<summary>TRL</summary>

[![TRL：严格的 L3 丢弃]({{ '/assets/blog/async-rl-staleness-2/sweep_trl-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_trl-blog.png' | relative_url }})

TRL 的核心参数是 `max_staleness`，它在 L3 的 [`RolloutQueueDataset`](https://github.com/huggingface/trl/blob/9d76f2f8c0421fcf8a4a72113a9e226ce796e592/trl/experimental/async_grpo/async_grpo_trainer.py#L367-L372) 里直接和每条 trajectory 的版本差比较，超过就丢弃。此外，TRL 还有个参数 `max_inflight_tasks` 来控制 L1 准入的容量，我们保持其为默认值 `-1`。此时，系统会自动推导 L1 准入的容量，将其设定为 [`max(max_staleness, 1)`](https://github.com/huggingface/trl/blob/9d76f2f8c0421fcf8a4a72113a9e226ce796e592/trl/experimental/async_grpo/async_grpo_config.py#L121-L128)。

从图中，我们可以看到，扫描 `max_staleness` 的六个点时，均未产生超限。实际最大 staleness 从 $$0$$ 到 $$5$$，逐点等于 `max_staleness` 的参数值。

不过，可以看到，扫描范围内的 waste 始终不为零：参数为 $$0$$ 时为 $$59.8\%$$，参数为 $$1$$ 时降至 $$22.2\%$$，放宽到 $$5$$ 后仍有 $$14.8\%$$。此外，单纯增大 `max_staleness` 参数也没有带来单调的吞吐提升：全量生成吞吐从 $$2189$$ 降至 $$1869$$ tokens/s，有效吞吐则先从参数为 $$1$$ 时的 $$1612$$ 降至参数为 $$3$$ 时的 $$1511$$，随后逐步回升至 $$1592$$。

这是因为 `max_staleness` 这个参数同时决定了 L1 的准入容量：容量加深让更多请求同时驻留，抢占和重新 prefill 随之增加，而这部分工作占用了 rollout 能力，全量吞吐因此缓慢下降。有效吞吐先降后升，则是因为 waste 同时在减少。因此总的来看，增大 staleness 上限，并不一定意味着能够获得更高的吞吐。

</details>

<details markdown="1" class="framework-detail" id="framework-areal">
<summary>AReaL</summary>

[![AReaL：只靠 L1 额度桶]({{ '/assets/blog/async-rl-staleness-2/sweep_areal-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_areal-blog.png' | relative_url }})

AReaL 的核心参数是 `max_head_offpolicyness`。在它的 [`StalenessManager`](https://github.com/areal-project/AReaL/blob/6cad8710bb5740a5480f8fba3a5970dd08638d01/areal/infra/staleness_manager.py#L100-L113) 里，这个参数只用来调整准入额度。

参数为 $$0$$ 时，系统处于完全同步状态，实际最大 staleness 为 $$0$$。而参数从 $$1$$ 增加到 $$5$$ 时，实际最大 staleness 依次为 $$3$$、$$4$$、$$6$$、$$6$$、$$8$$，**五个配置都超过了预期的 staleness 上限**，因此不满足严格的 staleness 约束。

此外，与 TRL 观察到的结果类似，吞吐并没有随额度放宽而提高。参数为 $$1$$ 时，有效吞吐达到 $$1916$$ tokens/s。后续几档则在 $$1787$$ 到 $$1848$$ tokens/s 之间变化，最大 staleness 却逐步增加到 $$8$$。在这份负载下，更大的准入额度带来了更旧的数据，但没有带来额外的吞吐收益。

</details>

<details markdown="1" class="framework-detail" id="framework-skyrl">
<summary>SkyRL</summary>

[![SkyRL：配额加违规计数，不丢弃]({{ '/assets/blog/async-rl-staleness-2/sweep_skyrl-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_skyrl-blog.png' | relative_url }})

SkyRL 的核心参数 `max_staleness_steps` 沿用 AReaL 的额度思路，只多了一处：取样时有一个[超限告警和计数](https://github.com/NovaSky-AI/SkyRL/blob/38146ee9248eb61d884e7df7b44747d97a6fc381/skyrl/train/fully_async_trainer.py#L1037-L1070)。告警能够记录问题，却不会阻止超限数据进入训练。因此，SkyRL 的最终表现与 AReaL 基本一致（SkyRL 的 partial rollout 默认保留上个版本的 KV，不需要重新计算即可原地继续，因此系统开销略有区别）。

</details>

<details markdown="1" class="framework-detail" id="framework-openrlhf">
<summary>OpenRLHF</summary>

[![OpenRLHF：以 batch 为粒度的许可]({{ '/assets/blog/async-rl-staleness-2/sweep_openrlhf-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_openrlhf-blog.png' | relative_url }})

OpenRLHF 的核心参数是 `async_queue_size`，可以理解为以 batch 为粒度的一组生成许可，训练侧[取走一批结果时才归还](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_trainer_async.py#L227-L232)。

参数为 $$1$$ 时，系统处于同步基线，实际最大 staleness 为 $$0$$。参数从 $$2$$ 增加到 $$5$$，实际最大 staleness 依次为 $$3$$、$$4$$、$$6$$、$$6$$，四个点都超过了参数值本身，与 AReaL 以及 SkyRL 的结论类似。队列长度只约束在途工作量，不能当作逐 token 的 staleness 上限。

</details>

<details markdown="1" class="framework-detail" id="framework-slime">
<summary>slime</summary>

[![slime：预取深度不是 staleness 上限]({{ '/assets/blog/async-rl-staleness-2/sweep_slime-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_slime-blog.png' | relative_url }})

slime 的核心参数是 `fully_async_group_prefetch_factor`。L1 阈值由各引擎并发参数 `sglang_server_concurrency` 之和除以 `n_samples_per_prompt`，再乘以该预取系数[得到](https://github.com/THUDM/slime/blob/2f4280e3b0e6934b954db93153bd79d9c9312830/slime/rollout/fully_async_rollout.py#L62-L103)。

本文只扫描预取系数 `fully_async_group_prefetch_factor`，`n_samples_per_prompt` 为算法侧固定设置的 $$8$$，同时每个引擎的 `sglang_server_concurrency` 固定为 $$64$$。这里让并发预算随 batch size 缩放，在我们的配置下，两个引擎的基数为 $$64\times2\div8=16$$ 个 prompt，正好是一个 batch，因此预取系数直接对应允许预取多少批。

从图中可以看到，参数依次取 $$1$$ 到 $$5$$ 时，对应的实际最大 staleness 为 $$3$$、$$6$$、$$5$$、$$6$$、$$7$$，全部高于系数 $$1$$ 到 $$5$$ 对应的预期值。与此同时，有效吞吐从 $$2020$$ 降至 $$1774$$ tokens/s。在这份负载下，加深预取只增加了 staleness，没有提高吞吐。

</details>

<details markdown="1" class="framework-detail" id="framework-roll">
<summary>ROLL</summary>

[![ROLL：一个参数贯穿 L1、L2、L3]({{ '/assets/blog/async-rl-staleness-2/sweep_roll-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_roll-blog.png' | relative_url }})

ROLL 的核心参数 `async_generation_ratio` 同时决定 L1 的准入额度、L2 的[运行中取消](https://github.com/alibaba/ROLL/blob/d96d0cce56ec42c78c7b61e111b596a3a3a008a8/roll/distributed/scheduler/generate_scheduler.py#L568-L577)和 L3 的[取样范围](https://github.com/alibaba/ROLL/blob/d96d0cce56ec42c78c7b61e111b596a3a3a008a8/roll/distributed/scheduler/generate_scheduler.py#L351-L386)。可以看到六个配置均未超限，实际最大 staleness 从 $$0$$ 到 $$5$$，逐点等于参数值。

随着 staleness 的增大，waste 从参数为 $$1$$ 时的 $$29.7\%$$ 降至参数为 $$5$$ 时的 $$14.2\%$$。所有异步设置都存在浪费，且主要来自 L2 取消。L3 过滤只占较小部分，这说明 ROLL 在生成过程中就停止了大部分无法满足版本限制的 trajectory。此外，全量生成吞吐从 $$2139$$ 降至 $$1912$$ tokens/s，有效吞吐则从 $$1504$$ 逐步增加，在参数为 $$4$$ 时达到 $$1643$$ tokens/s，随后基本不再变化。

</details>

<details markdown="1" class="framework-detail" id="framework-prime-rl">
<summary>prime-rl</summary>

[![prime-rl：max_off_policy_steps 扫描，L2 取消与 L3 过滤下的吞吐和实际 staleness]({{ '/assets/blog/async-rl-staleness-2/sweep_prime_rl-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_prime_rl-blog.png' | relative_url }})

prime-rl 的核心参数 `max_off_policy_steps` 同时用于 L2 和 L3。[min_fresh_version()](https://github.com/PrimeIntellect-ai/prime-rl/blob/d61df5690af1ccc2019c844403f8d2f2c5ed3273/src/prime_rl/orchestrator/utils.py#L41-L44) 根据该参数计算允许的最早生成版本：发布新权重时，系统据此取消仍在生成的 group。组装训练 batch 前，再用同一限制条件过滤待训练池。也就是说，调整这一个参数会同时影响 L2 与 L3 两层控制。

除了核心参数 `max_off_policy_steps` 外，参数 `max_inflight` 则保持为默认值 $$1024$$。它只是自适应并发控制器的上限，实际并发仍随 KV 使用情况调整。

所有配置下，实际最大 staleness 均能受控：`max_off_policy_steps` 为 $$1$$ 到 $$4$$ 时，最大 staleness 与 `max_off_policy_steps` 严格相等，继续放宽参数到 $$5$$，则不再产生更旧的样本，staleness 上限稳定维持在 $$4$$。随着上限放宽，waste 从参数为 $$1$$ 时的 $$34.4\%$$ 降到参数为 $$2$$ 时的 $$14.4\%$$，参数为 $$3$$ 时只剩 $$1.78\%$$，参数为 $$4$$ 起彻底降为零。有效吞吐从 $$1405$$ 升到 $$1749$$，再到 $$1865$$ tokens/s，参数为 $$4$$ 时达到 $$1897$$ 并饱和。参数为 $$1$$、$$2$$ 的浪费同时来自 L2 取消和 L3 过滤，且主要集中在 L2（L2 分别为 $$4.28$$M 和 $$1.73$$M token，L3 分别为 $$1.79$$M 和 $$0.33$$M token）。参数为 $$3$$ 只剩 $$0.24$$M 的 L2 取消，参数为 $$4$$ 起则彻底降为零。

</details>

<details markdown="1" class="framework-detail" id="framework-dressage">
<summary>Dressage</summary>

[![Dressage：按末尾 token 版本过滤]({{ '/assets/blog/async-rl-staleness-2/sweep_dressage-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_dressage-blog.png' | relative_url }})

Dressage 的核心参数 `dressage_staleness_keep_versions` 决定 L3 保留多少种末尾版本。它按不同版本首次出现的顺序记录版本，并依据末尾 token 的生成版本过滤，详见[版本记录与过滤逻辑](https://github.com/Accio-Lab/Dressage/blob/3e3142fe8ea07e4504c3b20a936a4c201a3de44c/dressage/rollout/staleness.py#L32-L132)。因此，参数设为 $$2$$，并不代表所有 token 的 staleness 都不超过 $$2$$。

本文只扫描 `dressage_staleness_keep_versions`，其余控制参数保持默认设置：`max_active_groups` 随 batch size 取 $$16$$，`max_partial_rollout_preempts=None`（表示未启用 L2 的版本切换次数限制）。

从图中我们可以看到，参数为 $$1$$ 时，waste 为 $$31.1\%$$。尽管丢弃了部分结果，但末尾 token 的版本不能覆盖前缀中更旧的 token，因此按照我们的定义，其并不能严格控制住 staleness，该参数配置下实测最大 staleness 为 $$3$$，超过了参数值本身。而参数放宽到 $$2$$ 时，waste 降为零。参数继续从 $$3$$ 放宽到 $$5$$，实际最大 staleness 都保持在 $$3$$，并维持吞吐稳定。

</details>

<details markdown="1" class="framework-detail" id="framework-nemo-rl">
<summary>NeMo-RL</summary>

NeMo-RL 提供四种 sampler，详见[代码实现](https://github.com/NVIDIA-NeMo/RL/blob/726d5eb225a8ae21e14b6778e56abfab2d794513/nemo_rl/algorithms/async_utils/staleness_sampler.py#L361-L712)。下面依次看 `windowed`、`weight_fifo`、`ready_first` 和 `in_order`。

本文在前三种模式中扫描 `max_staleness_versions`，在 `in_order` 中扫描 `max_lookahead_versions`。四种模式的 `max_inflight_prompts` 以及 `max_buffered_rollouts` 都固定为 $$96$$，为了确保扫描到 staleness 上限是 $$5$$ 的时候，可以预留出 $$16\times(5+1)=96$$ 个位置（参考[容量校验](https://github.com/NVIDIA-NeMo/RL/blob/726d5eb225a8ae21e14b6778e56abfab2d794513/nemo_rl/algorithms/single_controller_utils/config.py#L1375-L1384)）。

[![NeMo-RL windowed：L2 取消加 L3 过滤]({{ '/assets/blog/async-rl-staleness-2/sweep_nemo_windowed-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_nemo_windowed-blog.png' | relative_url }})

`windowed` 的 `max_staleness_versions` 同时用于 L2 和 L3，设计上限为参数本身。参数从 $$0$$ 放宽到 $$5$$，实际最大 staleness 依次为 $$0$$ 到 $$5$$，均未超限，waste 则分别为 $$16.4\%$$、$$24.0\%$$、$$30.1\%$$、$$27.6\%$$、$$23.5\%$$、$$13.2\%$$。waste 在参数 $$2$$ 处达到峰值，此后随窗口放宽而回落：窗口越宽，提前生成的数据越可能仍在窗口内被消费。

`weight_fifo` 除了限制 L1 准入，还会控制训练前的版本：在所有尚未消费的 group 里（还在跑的也算）找最老的版本，只从这个版本取，凑不满一批就等。下面是它的实验结果：

[![NeMo-RL weight_fifo：只等窗口内最旧的一版]({{ '/assets/blog/async-rl-staleness-2/sweep_nemo_weight_fifo-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_nemo_weight_fifo-blog.png' | relative_url }})

图中所有点都未超限，实际最大 staleness 等于参数取值。参数为 $$1$$ 到 $$5$$ 时，有效吞吐依次为 $$1825$$、$$1748$$、$$1760$$、$$1764$$、$$1778$$ tokens/s。`weight_fifo` 之所以能守住上限，靠的是训练侧停等：设参数取值为 $$\eta$$，准入最多允许 $$\eta+1$$ 批在途，取样时只取最老的一批，凑不满就停下来等。每个训练版本消耗一批，最老的一批最多排在 $$\eta$$ 批之后，因此 staleness 不会超过 $$\eta$$。可见，它只停等，不丢弃数据。

[![NeMo-RL ready_first：只有 L1 准入，不按版本过滤]({{ '/assets/blog/async-rl-staleness-2/sweep_nemo_ready_first-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_nemo_ready_first-blog.png' | relative_url }})

另外两种 sampler 中，`ready_first` 仅限制 L1 准入，不按版本过滤。参数为 $$0$$ 时为同步，参数从 $$1$$ 到 $$5$$ 时，实际最大 staleness 依次为 $$3$$、$$4$$、$$6$$、$$6$$、$$8$$，五个点均超限，表现与 AReaL、SkyRL 类似。

[![NeMo-RL in_order：准入时指定训练版本，不丢弃]({{ '/assets/blog/async-rl-staleness-2/sweep_nemo_in_order-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/sweep_nemo_in_order-blog.png' | relative_url }})

`in_order` 则通过 `max_lookahead_versions` 在准入时为每批数据指定目标训练版本。准入游标允许 lookahead 个批次领先训练，因此 staleness 上限等于参数：参数从 $$0$$ 到 $$5$$，对应的上限也是 $$0$$ 到 $$5$$，且不会产生任何 waste。可见，提前安排训练轮次，确实可以在不丢弃数据的情况下控制 staleness。不过，固定训练轮次也意味着，当前 batch 尚未凑齐时，训练侧需要等待。从图中可以看到，参数 $$1$$ 到 $$5$$ 的有效吞吐依次为 $$1740$$、$$1771$$、$$1780$$、$$1744$$、$$1761$$ tokens/s，放宽参数并不会带来明显的吞吐提升。

`in_order` 的控制方式与 `weight_fifo` 非常接近。稳态下每个版本只有一批，两者都按准入顺序消费，没准备好就等。区别只在分组单位：`weight_fifo` 按准入时的权重版本分组，`in_order` 则在准入时就把每个批次固定绑定到一个训练步。这一差别只在预热阶段才会显现，因为只有预热时多个批次才会共享同一个权重版本。此时，`weight_fifo` 会把多个批次当作同一组，连续用掉几个训练步，而 `in_order` 则要求严格绑定训练步。

</details>

## Part III：讨论

上面的实验给出了各套机制的运行结果。在此基础上，我们想进一步探讨如下几个问题：仅靠 L1 准入时，超限在训练中有多普遍？放宽准入为什么不一定更快？L2 取消与 L3 过滤各自改变了什么？staleness 超限丢弃是否会改变训练数据的构成？在相同的 staleness 上限参数下，哪套机制能给出更高的吞吐？下面我们将依次展开。

### 5. 仅靠准入额度，超限有多普遍？

在前面的实验中，AReaL、SkyRL 与 NeMo-RL 的 `ready_first` 等都出现了超限，但实验结果图只给出了整次运行的宏观统计量。这里我们更进一步，逐训练 batch 统计超限出现的频率和覆盖面。下面以 AReaL 为例，取它最紧的异步设置 `max_head_offpolicyness=1`。

[![仅 L1 准入时，每个训练 batch 的 staleness 构成]({{ '/assets/blog/async-rl-staleness-2/l1_violations-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/l1_violations-blog.png' | relative_url }})

图中，每根柱子对应一个训练 batch，按 staleness 分层，斜线阴影标出超过参数值的 trajectory。我们不难发现，**超限并非少数离群值**：$$50$$ 个 batch 中有 $$46$$ 个至少含一条超限 trajectory，$$6400$$ 条 trajectory 中累计 $$1280$$ 条超限。只有 $$4$$ 个 batch 完全没有超限。超限的 trajectory 中，$$1136$$ 条的 staleness 为 $$2$$，$$144$$ 条为 $$3$$。每个 batch 始终包含 $$128$$ 条 trajectory，这些超限数据并没有被过滤，全部实际参与了训练。因此，仅靠 L1 数量限制时，超限会持续出现在整个训练过程中，而非长尾导致的偶发现象。可以预见，这会对 RL 训练产生不可忽视且持续的影响。

### 6. 准入量越大，吞吐就越高吗？

**增加准入额度可以提高并发，但不保证吞吐持续上升。** AReaL 与 SkyRL 都在参数为 $$1$$ 时达到吞吐的峰值。此后，继续增大 staleness 上限，AReaL 吞吐从 $$1916$$ 降到参数为 $$5$$ 时的 $$1787$$ tokens/s，下降 $$6.7\%$$。而 SkyRL 则下降 $$4.0\%$$。因此，继续扩大准入则未必有效。TRL 同时调整准入与版本上限，有效吞吐也不单调：参数从 $$1$$ 到 $$2$$ 时先下降 $$6.2\%$$，随后才随 waste 减少而回升。

为了了解这一变化，我们统计了每条 trajectory 的抢占次数和重新 prefill 量。参数从 $$1$$ 增加到 $$2$$ 时，AReaL 的平均抢占次数从 $$0.34$$ 跳到 $$4.64$$，之后在约 $$6$$ 次附近变化。重新 prefill 量则继续上升，AReaL 从每条 $$5268$$ token 增至 $$18094$$，SkyRL 从 $$1744$$ 增至 $$9417$$。这意味着更多请求同时驻留会加大 KV 压力，被抢占的请求恢复时需要重算上下文，抵消并发带来的收益。

[![AReaL 与 SkyRL 随参数加深的抢占次数与重新 prefill 量]({{ '/assets/blog/async-rl-staleness-2/preempt_reprefill-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/preempt_reprefill-blog.png' | relative_url }})

**准入和 staleness 阈值是否可以独立调整，也会影响结果。** verl `drop` 分别使用 `num_warmup_batches` 与 `max_off_policy_threshold` 两个参数来调节准入与 staleness 阈值，TRL 则由一个 `max_staleness` 同时决定。下图在相同的预期 staleness 上限下比较两者：TRL 的 `max_staleness` 就是预期上限，verl 的阈值 $$\eta$$ 对应预期上限 $$\eta-1$$。我们在同一 staleness 上限下，对比二者的有效吞吐以及 waste 占比。

[![相同预期上限下，TRL 与 verl drop 的有效吞吐和 waste]({{ '/assets/blog/async-rl-staleness-2/trl_verl_matched-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/trl_verl_matched-blog.png' | relative_url }})

从图中我们可以看到，预期上限为 $$0$$、$$1$$ 时，TRL 的有效吞吐较高，分别为 $$880$$ 和 $$1612$$ tokens/s，verl 为 $$838$$ 和 $$1436$$。为 $$2$$、$$3$$、$$4$$、$$5$$ 时，verl 较高，分别为 $$1816$$、$$1945$$、$$1945$$、$$1951$$，TRL 为 $$1513$$、$$1511$$、$$1563$$、$$1592$$。verl 在预期上限为 $$4$$、$$5$$ 时实际都只达到 staleness $$3$$，因此吞吐在 $$3$$ 之后基本走平，waste 维持为 $$0$$，而 TRL 则一路升到上限 $$5$$。这是因为 verl 可以独立限制预取量，将其控制在较低值，从而减少 waste。反观 TRL，其预取量也随 staleness 增大而膨胀，因此相应的 waste 在已测范围内无法降到零。

### 7. 运行时取消（L2）与训练取样时过滤（L3）的影响是什么样的？

为了进一步分析 L2 取消与 L3 过滤的作用，我们额外拼接了一个消融配置：L1 采用 AReaL 的额度桶，L2 采用 ROLL 的运行时取消，L3 采用 verl `drop` 的训练前过滤。L1-L3 三层共用同一个参数 $$\eta$$：L1 的准入上限由 $$\eta$$ 控制。L2 在每次发布新权重时，取消版本差已经超过 $$\eta$$、且仍在生成的 group。L3 则在训练前过滤掉版本差超过 $$\eta$$ 的数据。下图展示了逐级叠加 L1-L3 的结果对比。

[![同一准入下逐级叠加 L3 与 L2：吞吐、丢弃来源与实际 staleness]({{ '/assets/blog/async-rl-staleness-2/ladder_l2_l3-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/ladder_l2_l3-blog.png' | relative_url }})

**L1：** 参数 $$\eta$$ 从 $$1$$ 增至 $$5$$，实际最大 staleness 分别为 $$3$$、$$4$$、$$6$$、$$6$$、$$8$$，均超过 $$\eta$$。

**L1+L2：** 参数 $$\eta$$ 从 $$1$$ 增至 $$5$$，实际最大 staleness 为 $$2$$、$$3$$、$$4$$、$$5$$、$$6$$，始终比预期多 $$1$$ 个版本。相比只有 L1，其能在一定程度上降低最大 staleness，但仍没法严格控制。这是因为 L2 只在发布新权重时取消还在生成、且版本差已经超过 $$\eta$$ 的 group。已经完成、但版本差超过 $$\eta$$ 的 group 不会被取消，仍然会进入训练。

**L1+L3：** 各点的实际最大 staleness 得以控制，并严格等于 $$\eta$$。但代价是吞吐显著降低：有效吞吐降至 $$1417$$–$$1578$$ tokens/s，较仅 L1 的总吞吐下降了 $$12\%$$–$$26\%$$。这主要是因为有 $$14.2\%$$–$$27.5\%$$ 的生成 token 被丢弃。

**L1+L2+L3：** 实际最大 staleness 仍严格等于 $$\eta$$，但除 $$\eta=2$$ 基本持平外，有效吞吐较 L1+L3 提高 $$1\%$$–$$7\%$$。这是因为，启用 L2 后，一部分丢弃由 L3 的“完成后过滤”前移到了 L2 的“运行中取消”（图中阴影部分的不同占比说明了这一点）。可见，在 L1+L3 的前提下，L2 能带来额外收益：发布新权重时，那些已注定超限、却仍在生成的 group 会被立即取消，不再占用 rollout 容量。训练也因此无需等它们生成完成后再丢弃。

不过，在 L3 的基础上增加 L2，waste 的比例并没有下降，反而从 $$14.2\%$$–$$27.5\%$$ 略升至 $$14.5\%$$–$$29.8\%$$。其背后的原因是，被释放的容量会继续生成替代数据，而其中一部分之后同样会被丢弃。由于 L2 丢弃得更早，最终整体丢弃量反而可能更多。因此，L2 虽然缩短了训练等待，却没有减少总浪费。

总体而言，在这组消融实验中，L3 负责在训练取样时守住 staleness 上限，L2 则负责提前终止注定作废的生成。没有 L3 时，L2 只能取消运行中的 group，已完成的超限数据仍会进入训练，因此 **L3 才是 staleness 严格控制的关键**。L2 在此基础上提前停止无效生成，释放 rollout 容量，可以带来额外的吞吐收益，但不保证整体 waste 比例下降。

### 8. 丢弃会偏向长 trajectory 吗？

实验中，我们用 waste 描述浪费了多少 token，却没有进一步分析：**最后留给训练的，究竟是哪一部分数据？** 对按 trajectory staleness 丢弃的机制，例如 verl `drop`、TRL 和 ROLL，response 越长，生成期间越可能跨过更多训练版本，也就越容易超限。因此，被丢弃的 trajectory 可能系统性地偏长。

为检验这一机制，我们进一步引入 Dressage 作为对照：Dressage 依据末尾 token 的生成版本过滤，而不是直接检查整条 trajectory 已跨过多少训练版本。如果上述解释成立，那么按整条 trajectory 的 staleness 丢弃的常规方案应表现出更明显的长度偏向，而 Dressage 的这种偏向应较弱。

[![被丢弃的 trajectory 的长度分布，以及各框架最长 10% 中被丢弃的占比]({{ '/assets/blog/async-rl-staleness-2/length_bias-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/length_bias-blog.png' | relative_url }})

左图以 verl `drop`、`max_off_policy_threshold=1`、`num_warmup_batches=4` 为例：实色柱表示进入训练的 $$1280$$ 条 trajectory，阴影柱表示被丢弃的 $$4512$$ 条，两条虚线标出各自均长。可以看到，被丢弃集合的均值更高，在 $$20000$$ token 以后仍有明显长尾。相比之下，实际进入训练的集合在该区间已经很少。

右图进一步跨框架比较：横轴为框架，同一框架内按实际达到的最大 staleness 分层，柱高表示最长 $$10\%$$ 的 trajectory 中被丢弃的比例，黑色短横线表示整体丢弃率，点状空心柱为 Dressage 对照。

| 配置 | 全部 trajectory 中被丢弃的占比 | 最长 $$10\%$$ 中被丢弃的占比 | 富集倍数 | 被丢弃均长 / 被消耗均长 |
|---|---|---|---|---|
| verl `drop`（staleness $$1$$–$$2$$） | $$2.4\%$$–$$13.0\%$$ | $$21.4\%$$–$$71.4\%$$ | $$5.48$$–$$8.76$$ | $$2.42$$–$$3.03$$ |
| TRL（staleness $$1$$–$$5$$） | $$7.5\%$$–$$12.1\%$$ | $$38.4\%$$–$$60.8\%$$ | $$4.50$$–$$5.73$$ | $$2.14$$–$$2.40$$ |
| ROLL（staleness $$1$$–$$5$$） | $$7.5\%$$–$$14.9\%$$ | $$40.6\%$$–$$82.7\%$$ | $$4.30$$–$$5.61$$ | $$2.20$$–$$2.52$$ |
| Dressage（staleness $$3$$） | $$28.3\%$$ | $$44.4\%$$ | $$1.57$$ | $$1.15$$ |

表中仅列出各框架存在丢弃的配置，并标注其覆盖的实际 staleness。verl `drop` 仅在 staleness 为 $$1$$、$$2$$ 时有丢弃。继续放宽后，最优点已不再丢弃数据。Dressage 仅在参数取 $$1$$ 时放弃部分数据，此时实际 staleness 为 $$3$$。

富集倍数定义为最长 $$10\%$$ 的丢弃率除以整体丢弃率，用于衡量长 trajectory 相对一般 trajectory 的丢弃倾向。所有按整条 trajectory staleness 丢弃的配置，富集倍数均大于 $$1$$，说明长 trajectory 确实更易被丢弃。例如，verl `drop` 在 staleness 为 $$2$$ 时整体只丢弃 $$2.4\%$$，最长 $$10\%$$ 中丢弃 $$21.4\%$$，富集倍数高达 $$8.76$$。

从实验结果中我们还可以发现，随着 staleness 上限放宽，富集倍数呈增长趋势，这说明**丢弃越来越集中于长 trajectory**。这与按 staleness 过滤的逻辑一致：上限收得很紧时，各种长度的数据都可能被过滤。上限放宽后，仍然超限的往往主要是耗时最长的请求。

Dressage 作为对照，表现出明显更弱的长度偏向。参数为 $$1$$ 时，其丢弃与保留集合的均长比为 $$1.15$$，最长 $$10\%$$ 的丢弃率为 $$44.4\%$$，整体为 $$28.3\%$$，富集倍数为 $$1.57$$，远低于其余框架按整条 trajectory staleness 丢弃的方案。这与预期一致：Dressage 依据末尾 token 的生成版本过滤，而不是直接检查整条 trajectory 已跨过多少版本，因此不会像按整条 trajectory staleness 丢弃的方案那样系统性地偏向长 trajectory。

可见，丢弃的代价不单纯是计算浪费，还包括训练数据构成的变化。对推理任务而言，较长的 response 可能对应更难的问题。如果按 staleness 丢弃系统性地偏向长 trajectory，模型就可能少学到一部分较难样本，进而影响收敛。这一影响是否显著，仍需核对 trace 中长度与难度的关系，并测量这种偏向对训练效果的实际影响。

总的来看，**丢弃带来了计算浪费，也带来了训练数据构成的变化**。这种数据构成变化对 RL 训练的影响可能比预想更复杂，也更难预先判断。因此，类似 NeMo-RL `weight_fifo` 以及 `in_order` 的控制机制其实值得更多关注与研究：它能在保证 staleness 的同时不产生任何丢弃，从而避免对数据构成的破坏。这也引出了我们后续第三篇博客的探讨。

### 9. 相同 staleness 上限下的吞吐对比如何？

实际配置系统时，我们通常先确定能够接受的 staleness，再寻找这一限制下吞吐较高的参数。因此，我们这里可以做一个不同 RL 框架间相对公平的比较：给定 staleness 上限参数 $$\eta$$，各套机制在不超过该上限时，能提供多少有效吞吐？

具体来说，我们对每个 $$\eta$$，选出在所有**实际最大 staleness 不超过 $$\eta$$** 的运行中，有效吞吐最高的配置，来进行对比。这里采用“不超过”，意味着更紧的设置（实际 staleness 更小）在更大的 $$\eta$$ 下仍然合格，可以继续成为最优点。按此方法，最优吞吐随 $$\eta$$ 增大必然单调不降。如果曲线不再上升，就说明在已扫描的参数范围内，继续增大 $$\eta$$ 已无额外收益。

> **说明**：跨权重更新时（partial rollout 时），部分框架会保留旧 token 对应的 KV cache。按本文引用的版本，SkyRL 的 fully-async 路径和 prime-rl 默认会保留旧 KV，避免权重更新引起的 prefix 重算，这会一定程度上增大其吞吐。

[![各框架在相同 staleness 上限参数下的最高有效吞吐]({{ '/assets/blog/async-rl-staleness-2/matched_throughput-blog.png' | relative_url }})]({{ '/assets/blog/async-rl-staleness-2/matched_throughput-blog.png' | relative_url }})

从图中我们可以发现，$$14$$ 个配置中有 $$11$$ 个从参数 $$\eta=3$$ 起不再变化，除了 SkyRL、ROLL 和 NeMo-RL `windowed`：SkyRL 在 $$\eta=4$$ 时还有一次提升，ROLL 和 NeMo-RL `windowed` 则继续上升到 $$\eta=5$$。也就是说，**在当前负载、资源配比及扫描范围内，多数配置在 staleness 上限参数 $$\eta$$ 达到 $$2$$ 或 $$3$$ 后，已经无法通过继续增大参数获得更高吞吐。**

首先，我们可以发现，当 staleness 上限较小时，在严格的 staleness 约束下，部分框架甚至没有可用配置（我们不考虑彻底同步的配置）。如果算法侧只能容忍很紧的 staleness，例如要求实际最大 staleness 不超过 $$1$$ 或 $$2$$，那么此时，就基本只能选择那些具有严格 staleness 控制的框架。例如，如果要求 $$\eta=1$$，那么 AReaL、SkyRL、slime、OpenRLHF、Dressage 以及 NeMo-RL `ready_first` 均无法使用，因为其最小实际 staleness 是 $$3$$。能采用的是 verl `drop`、verl `wait`、TRL、ROLL、prime-rl，以及 NeMo-RL 的 `windowed`、`in_order` 和 `weight_fifo`。在这些框架中，我们不难发现，不涉及 L3 丢弃的 NeMo-RL `in_order` 和 `weight_fifo` 很有竞争力：$$\eta=1$$ 时 `weight_fifo` 为 $$1826$$、`in_order` 为 $$1744$$，都高于同样具有严格 staleness 控制的 TRL（$$1612$$）、ROLL（$$1512$$）、NeMo-RL `windowed`（$$1465$$）、verl `drop`（$$1438$$）和 prime-rl（$$1405$$）。而 $$\eta=2$$ 时 NeMo-RL 的 `weight_fifo` 为 $$1826$$、`in_order` 为 $$1777$$，与 verl `drop` 的 $$1824$$ 基本持平，并且同样优于其余严格控制 staleness 的方案。**可见，在严格容忍度下，用等待换不丢弃并没有牺牲吞吐，反而往往比在 L1 准入时不加考虑、全部交给 L3 丢弃效果更好。**

其次，若算法侧对 staleness 容忍度较高，infra 选型的重点便转向上限放宽后的绝对系统吞吐。此时领先的是 SkyRL（$$2053$$ tokens/s），slime 与 Dressage 紧随其后（都是 $$2012$$），再往下是 AReaL（$$1959$$）以及 verl 与 OpenRLHF（$$1951$$）。而 TRL、ROLL 和 NeMo-RL `windowed` 等其余方法则明显落后。我们发现，影响吞吐的一个重要因素，是框架能否把 **L1 容量参数和 staleness 上限参数解耦**。Dressage 的 `max_active_groups`、OpenRLHF 的 `async_queue_size`、slime 的 `fully_async_group_prefetch_factor`、verl 的 `num_warmup_batches` 都可以单独设置，与控制 staleness 上限的参数互不干扰。因此，这些框架可以在放宽 staleness 上限的同时把准入保持在较小额度，避免 KV 压满。如果框架采用 L3 丢弃（例如 verl `drop`），其甚至能让框架在高 staleness 上限下彻底避免 L3 丢弃。而 TRL 和 ROLL 则是一个参数同时决定准入容量和 staleness 上限，放宽 staleness 上限就顺带增加在途 trajectory 数目，导致更多的抢占、重新 prefill 以及 L3 丢弃，使得吞吐受损。

## Part IV：小结

在这篇博客中，我们基于自研的 RL 模拟器 [**SimRL**](https://github.com/psrl-project/SimRL)，测试了不同框架中 staleness 控制机制的具体表现，验证了我们在第一篇博客中的分析。此外，我们还补充了一些对比实验，得到了一些有意思的结论。本文的几个核心 takeaway 如下。

**参数名不等于保证。** 很多框架的参数名乍看与 staleness 上限相关，但实际往往只控制 L1 准入量或在途工作量，并不能真正约束 staleness 上限。

**超限是结构性的。** 仅靠 L1 准入控制时，慢 trajectory 可能被后续快 trajectory 反复绕过，直到多个训练版本之后才被取用。其带来的 staleness 超限并非少数离群值，而是在多数 batch 中反复出现。

**放宽 L1 准入可能反而带来吞吐负收益。** 在本文的负载和资源配比下，多数配置在 staleness 上限参数达到 $$2$$ 到 $$3$$ 之后，继续增大参数已经换不到吞吐。甚至，容量继续加大反而会让更多请求同时驻留，引发抢占和重新 prefill，抵消并发收益。

**L2 取消和 L3 过滤。** 实验表明，严格控制 staleness 的关键在 L3。L2 只能取消仍在生成中的超限数据。已经完成、但在等待训练时已超限的结果，仍可能进入训练。因此，只有 L2 是不够的，还需要在 L3 对实际进入训练的数据做版本检查，排除超限结果，才能真正守住上限。L2 的作用是在此基础上提前停止无效生成，提高 rollout 容量的利用效率，不过也并不保证整体 waste 比例下降。

**端到端性能收益来源。** 端到端性能收益主要来自两方面。首先是生成本身的效率，这需要一个合适的 L1 准入上限：准入太少，生成侧并发不足，难以充分利用 GPU。准入太多，又会加大 KV 压力，引发更多抢占和重新 prefill，反而降低生成效率。因此，准入上限需要结合负载和资源配比设置。另一个因素是生成结果的利用率。即使生成得很快，如果大量结果最终因 staleness 超限被丢弃，这部分计算也无法转化为有效吞吐。因此，最简单的提高端到端性能的方法，就是在放宽 staleness 上限的同时，适当维持住先前的预取量，从而兼顾生成效率与结果利用率。

**严格 staleness 控制的两条路。** 可以看到，大部分框架采用的严格控制 staleness 的方法，都是在 L3 训练前丢弃。但这一路线有两个明显弊端：（1）丢弃对于有效吞吐的损害比较严重，使其明显劣于其余不丢弃数据的框架。（2）这种丢弃是有偏的，较长的 trajectory 更容易被丢弃，从而改变训练实际使用的数据构成。而与之对应的，另一条严格控制 staleness 的路线是通过限制取样和批次安排来避免数据丢弃，典型代表是 NeMo-RL 的 `weight_fifo` 与 `in_order`。其本质是以等待避免浪费，较慢的长尾数据会直接推迟训练步。

考虑到目前严格控制 staleness 的方案要么依赖 L3 丢弃，损害有效吞吐且带来有偏的数据构成，要么通过限制取样和批次安排来避免丢弃，却可能因长尾等待而推迟训练步，那么一个很自然的问题就是：**我们能否在不丢弃数据、严格控制 staleness 的同时，给训练 batch 的组装保留更多灵活性，减少长尾等待？** 我们将在下一篇中尝试解答这个问题。

以上就是本篇博客的全部内容。欢迎关注我们的开源项目 [**PSRL**](https://github.com/psrl-project/psrl)，如果这个系列的博客对你有帮助，希望你能顺手给我们的项目点个 star⭐！
