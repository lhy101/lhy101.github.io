---
layout: "blog-post"
title: "异步 RL 框架的 Staleness，究竟怎么控制（三）？"
description: "我们设计了一套 Reserve–Occupy 协议，在严格控制 staleness 的前提下，不丢弃任何数据，并最大化有效 RL 训练吞吐。"
date: "2026-10-31"
lang: "zh-CN"
series: "PSRL / 03"
permalink: "/blog/psrl-staleness/"
cover: "/assets/blog/psrl-staleness/cover.svg?v=60fbd48e7a9b"
cover_alt: "Reserve–Occupy 协议：生成前预留合法位置，完成后尽早放置，并共同满足版本上限"
cover_width: 1280
cover_height: 720
author_profile: false
---

# 异步 RL 框架的 Staleness，究竟怎么控制（三）？

在系列博客的第一篇中，我们沿着一条 trajectory 的生命周期，详细探讨了不同异步 RL 框架的三层 staleness 控制：生成前的 L1 准入，生成过程中的 L2 控制，以及训练前的 L3 处理。文章最后留下了一个问题：

**我们能否设计一套更灵活的控制机制，既不丢弃数据，又能保证最大 staleness？**

让我们先回想一下目前保证最大 staleness 的常见做法：L3 丢弃。数据进入训练前，我们检查它的版本，发现太旧就不用了。这样的做法尽管守住了上限，但被丢弃数据的生成、后处理和 reward 计算早已完成，这些计算就白白浪费了。

而 NeMo-RL 的 `in_order` 则提供了另一个思路：既然不想等到最后才发现数据太旧，那就在数据准入时，提前安排好它要参加哪一轮训练。正常按序执行时，数据按约定的轮次取用，不必等到完成后再因为版本太旧而被淘汰。但这个办法也有不够灵活的地方。假如当前 batch 就差一条长尾，后面 batch 的几条数据却早已完成，由于目标轮次已经绑定，我们仍然不能拿后面的结果补上当前这个空缺。

那么，这个问题是不是可以进一步得到解决呢？

顺着这个思路想，提前安排训练位置确实可以不用丢弃数据就把 staleness 直接控制住，但我们一定要在一开始就把最终训练 batch 定死吗？我们能不能先为 trajectory 留好一个位置，等它真正完成时，再根据当时的情况调整？这样，快的 trajectory 就有机会早点参与训练，慢的 trajectory 也不至于因为一直被往后推而超限。

为此，我们提出了一套 *Reserve*–*Occupy* 协议，其核心想法是把**生成侧的准入**和**训练侧的 batch 组装**放在一起考虑，让两者在同一个协议下配合起来：准入前就考虑数据之后能在哪里训练，结果完成后的 batch 排布又反过来影响后续准入。这样，我们既能在不丢弃数据的情况下严格控制住最大 staleness，又能避免过早固定训练 batch，让调度更灵活。

下面，我们将抽丝剥茧，一步步讲解我们这套 staleness 控制协议。

## Part I：把 staleness 控制变成一个放置问题

### 1. 先给 trajectory 和 buffer 标上版本

异步训练里，一条数据生成出来时用的模型，和它后来参加训练时的模型，可能已经不是同一个版本了。要控制两者相差多远，首先得知道这条数据是用哪个版本生成的。

我们给每条 trajectory 记录一个 **$$v_{\text{traj}}$$**，表示生成它所用的权重版本。例如，用 $$v_0$$ 生成的数据就记为 $$v_{\text{traj}}=0$$。这一节先考虑一条 trajectory 从头到尾使用同一版模型的情况，涉及多个版本的 partial rollout 放到后面的扩展里讨论。

接下来再看训练侧。训练端每次取一个 batch 更新模型，我们就在 rollout 输出和训练端之间引入一层 **staleness buffer**，用它来安排每条数据进入哪一轮训练。这一层与别的框架中 L3 前的 queue/buffer 的设计类似，但不同的是，我们的 staleness buffer 会提前规划好训练版本：**我们会预设一些 staleness buffer，每个 staleness buffer 对应一个训练 batch，大小就是 batch size。** 比如 batch size 为 $$4$$，每个 buffer 就有 $$4$$ 个槽位，每个槽位容纳一条 trajectory。等 $$4$$ 条数据都准备好了，这一批才可以交给训练端去训练。

因此，我们为每个 buffer 分配一个版本号 **$$v_{\text{buffer}}$$**，表示这一批数据将用于哪个模型版本的训练，例如：

- **Buffer $$v_0$$** 用于模型从 $$v_0$$ 更新到 $$v_1$$ 的训练。
- **Buffer $$v_1$$** 用于模型从 $$v_1$$ 更新到 $$v_2$$ 的训练。
- **Buffer $$v_2$$** 用于模型从 $$v_2$$ 更新到 $$v_3$$ 的训练。

注意，Buffer $$v_1$$ 里存放的数据不一定由这一版模型生成：它只说明这些数据将一起参加 $$v_1\rightarrow v_2$$ 这轮更新。如果把一条 $$v_0$$ 生成的数据（即 $$v_{\text{traj}}=0$$）放进 Buffer $$v_1$$，它参加训练时的 staleness 就是 $$1-0=1$$。放进 Buffer $$v_2$$，staleness 就成了 $$2-0=2$$。

这样，**staleness 控制就被转化成了一个放置问题**。设最大允许 staleness 为 $$\eta$$，要控制住 staleness，我们只要让每条 trajectory 最终落入的 buffer 满足：

$$
\begin{equation*}
v_{\text{buffer}} \leq v_{\text{traj}}+\eta
\end{equation*}
$$

这个不等式给出了数据最晚能放置的那个 buffer。例如 $$v_{\text{traj}}=0$$、$$\eta=1$$，这条数据可以进入 Buffer $$v_0$$ 或 Buffer $$v_1$$，却不能再被安排到 Buffer $$v_2$$。Buffer $$v_2$$ 即使完全空着，对它来说也已经太晚了。

![把 staleness 控制转化为放置问题：每条数据的最终 buffer 必须满足版本上限]({{ '/assets/blog/psrl-staleness/staleness-buffer-concept.svg' | relative_url }})

图里的 $$X$$ 来自 $$v_0$$，放进 Buffer $$v_1$$ 时满足 $$1\leq 0+1$$。如果继续往后放进 Buffer $$v_2$$，就会违反 staleness 上限。

值得一提的是，这里的 buffer 只记录 trajectory 的元数据（包括 trajectory ID、版本等）。可以把不同 buffer 看成 trajectory 与训练之间的一张排布表，因此调整 trajectory 的位置不需要复制或移动实际的 trajectory 数据。

这一步的意义在于，版本边界可以在生成开始前就被检查。准入阶段只要找得到一个合法的 *Empty* 槽位，就能把这条 trajectory 纳入后续安排。之后即使它成为长尾，也不会因为完成得晚而突然失去唯一的合法去向。真正完成时，协议只是在这张排布表上重新选择槽位，数据本身仍沿着原来的 rollout 路径返回。

### 2. 有了放置规则，还需要哪几个操作？

那么，具体该怎么去放置呢？最直接的想法，就是等一条数据完成了再给它找合法的位置。但问题也在这里：假设有一条长尾的 trajectory，等它慢慢生成完，前面允许它放置的 buffer 可能已经满了，甚至可能都已经被拿去训练了。这时再找位置，就只能放在一个靠后的 buffer 中，不满足 $$v_{\text{buffer}} \leq v_{\text{traj}}+\eta$$ 的约束，从而超出 staleness 上限。

所以，我们需要把安排放置的时机拆成两个部分。**开始生成前，我们需要先预留一个合法位置，确保生成完成后仍有合法位置，不会超限。真正生成完成时，再决定最终放在哪里（最差也可以放在之前预留的位置上）。** 我们把前一步叫 *Reserve*，后一步叫 *Occupy*。

这也意味着，buffer 里会同时出现尚未完成和已经完成的 trajectory。为了区分它们，我们让每个槽位有 $$3$$ 种状态：

- *Empty*：还没有安排任何 trajectory 的空槽。
- *Reserved*：已经为一条 trajectory 预留位置，但结果还没完成。
- *Occupied*：trajectory 已经完成，实际放进了这个槽位。

有了这 $$3$$ 种状态，L1 准入检查就很好理解了：**看看现在还能不能为这条 trajectory 做一次 *Reserve*。** 我们扫描尚未取用、满足 $$v_{\text{buffer}}\leq v_{\text{traj}}+\eta$$ 的 buffer，检查其中是否还有 *Empty* 槽位。有的话，先把一个空槽变成这条 trajectory 的 *Reserved*，再开始生成。没有的话，就无法保证新准入的 trajectory 生成完毕后还有合法位置，因此必须暂缓准入。

![L1 准入：四条 trajectory 依次 Reserve 并开始生成，两个合法 buffer 满后暂缓准入]({{ '/assets/blog/psrl-staleness/reserve-admission.gif' | relative_url }})

*Reserved* 虽然还没有实际数据，但也不能再算成空槽，因为它已经为一条在途 trajectory 留好了位置。多个 trajectory 同时准入时，也必须以成功预留为准，不能因为各自都看到过空位，就一起开始生成。这样，每条成功准入的 trajectory 才确实有一个合法位置可去。

这样，我们就把生成侧和训练侧联系了起来。如果只看生成侧的数量，我们无法知道每条数据最晚要赶上哪一批训练。如果只看已经完成的数据，我们又无法知道后面还有多少旧数据正在生成。但通过生成前的 *Reserve* 提前规划，再通过生成后的 *Occupy* 实际安排训练批次，我们就可以将二者联系在一起了。

### 3. 按最差情况 *Reserve*，按最好情况 *Occupy*

那么，具体应该 *Reserve* 和 *Occupy* 到哪个槽位呢？

开始生成时，我们还不知道谁会先完成。既然如此，不妨先按最差情况来安排，**优先在合法范围内较晚的 buffer 中 *Reserve*，并优先使用其中靠后的空槽。** 即使这条 trajectory 很慢，它也已经有了一个不超过上限的最差位置。同时，前面的空槽可以尽量留给较早完成的数据，帮助更早的 batch 攒齐。

这里的“较晚”是一个优先级，不是要求一定放到 $$v_{\text{traj}}+\eta$$。例如 $$v_{\text{traj}}=0$$、$$\eta=1$$，我们优先预留 Buffer $$v_1$$。如果它已经满了，Buffer $$v_0$$ 还有空槽，就仍然可以在那里预留，不必直接拒绝准入。

等 trajectory 完成，最稳妥的做法当然是**就在自己当前的预留位置 *Occupy***。这个位置此前已经通过检查，直接把 *Reserved* 变成 *Occupied* 就可以保证不超限。不过如果始终这样做，生成前安排在哪个 batch，最终就留在哪个 batch，调度思路也就退回到类似 NeMo-RL `in_order` 的固定批次安排。

来看一个小例子。A–F 都由 $$v_0$$ 生成，$$\eta=1$$，每个 buffer 有 $$4$$ 个槽位。假设 B 最先完成，它原本预留在 Buffer $$v_1$$ 的第 $$3$$ 个槽位。按原地 *Occupy* 的做法，我们只需把这个槽位从 *Reserved* 变为 *Occupied*，其他位置完全不用动。

![原地 Occupy：直接使用预留位置]({{ '/assets/blog/psrl-staleness/occupy-in-place.gif' | relative_url }})

这样当然合法，但也有些可惜：B 明明已经完成，前面的 Buffer $$v_0$$ 也有空槽，它却只能留在后面的 Buffer $$v_1$$，等到下一轮才参与训练。

那么，真的必须 *Occupy* 在原先 *Reserve* 的位置吗？其实不必。*Reserve* 只提供“最差也有合法位置”的保证，而不是替 *Occupy* 固定最终放置位置。事实上，只要最终的放置位置依然满足

$$
\begin{equation*}
v_{\text{buffer}}\leq v_{\text{traj}}+\eta
\end{equation*}
$$

就可以把已经完成的数据放得更早。

具体来说，我们可以**释放原先的预留槽位，整理可移动的 *Reserved*，再寻找当前最早的合法 *Empty* 槽位，将结果放进去。** 这里的“最早”先按 buffer 版本比较，再按同一个 buffer 内的槽位顺序比较。

这里为什么还要整理其他 *Reserved*？因为刚释放的空位可能在后面的 buffer，而前面还留着一些正在生成的 trajectory 的预留。既然这些数据还没完成，只要版本允许，就可以把它们的预留后移，用上刚腾出来的空位，继续把靠前的空间留给已完成的数据。

回到刚才那个例子：B 预留在 Buffer $$v_1$$ 的第 $$3$$ 个槽位并率先完成。这一次，我们这样处理：

1. 删除 B 的 *Reserved*，把已经完成的 B 暂放到图中右上角的待放置区。
2. 从前往后扫描 *Reserved*，找到第一个能合法移入这个空位的 F。把 F 的预留移到 B 刚刚释放的 Buffer $$v_1$$ 第 $$3$$ 个槽位。
3. 为 B 寻找最早的合法 *Empty* 槽位。Buffer $$v_0$$ 第 $$1$$ 个槽位原本就是空的，因此 B 直接落在那里。

![按最早位置 Occupy：其余预留位置填充空位]({{ '/assets/blog/psrl-staleness/reserved-repacking.gif' | relative_url }})

和前一张图相比，B 从后面的 batch 提前到了前面的 batch，F 则接过了 B 原先的预留位置。这样调整是安全的，因为 B 和 F 各自都有一个满足版本上限的位置：B 已经完成，可以直接填进更早的 batch，而 F 虽然还在生成，但后移之后依然保有合法预留，并没有因为给 B 让出空间而失去位置。

**这样调整的目的，是让已经完成的数据尽早参与训练，同时让尚未完成的 trajectory 的预留尽量靠后。** 本例中，Buffer $$v_0$$ 第 $$1$$ 个槽位是空的，B 可以直接占用，帮助训练侧攒齐 batch。而后移 F 的作用，是把它占着的第 $$3$$ 个槽位也腾出来，给后续完成的数据留下更多靠前的空间。

当然，不是每个 *Reserved* 都可以随意后移。F 能不能填进这个空位，要看 F 自己的版本和目标 buffer 的版本。把目标空槽所属 buffer 的版本记作 $$v_{\text{gap}}$$，搬移必须满足

$$
\begin{equation*}
v_{\text{gap}}\leq v_{\text{traj}}(F)+\eta
\end{equation*}
$$

本例中就是 $$1\leq0+1$$，所以 F 可以移动。扫描时，我们选择前面符合条件的最早 *Reserved* 即可。

这就是“按最差情况 *Reserve*，按最好情况 *Occupy*”：**生成前先留好合法位置，完成后再根据当前排布尽量往前放。** 这里的“最好”是当前状态下尽早放置，不是预知所有完成时间后的全局最优。如果没有更早的空槽，最差也可以使用本次操作前仍有效的预留位置。这样，我们既保住了每条 trajectory 的 staleness 上限，又为前面的 batch 提前攒齐创造了条件。

这里的两个动作也各自承担不同责任。*Reserve* 面向还没有结果的 trajectory，优先保证“以后至少有一个合法位置”。*Occupy* 面向已经完成的 trajectory，利用当时已经释放出来的空槽，尽量让更早的 batch 先凑齐。两者共享同一个版本不等式，因此完成顺序可以变化，安全边界却不会变化。

![Reserve 按最差情况预留，Occupy 按当前情况尽早放置，两者共享同一条上限]({{ '/assets/blog/psrl-staleness/protocol-lifecycle.svg' | relative_url }})

### 4. Buffer 状态

按照这样的协议操作，最终每个 buffer 中一定是 *Occupied* 集中在前面，*Reserved* 留在后面，中间是 *Empty*。因此，实际的 buffer 只会出现下面 $$3$$ 种状态：

- *PENDING*：还没凑齐已完成的数据，但仍有空槽。
- *STUCK*：空槽已经用完，其中仍有 *Reserved*，要等未完成的 trajectory。
- *READY*：整批都是 *Occupied*，数据已经齐了。

![PENDING、STUCK、READY：有空槽、满但未齐、整批已齐]({{ '/assets/blog/psrl-staleness/buffer-states-feedback.svg' | relative_url }})

如果当前版本的 buffer 已经 *READY*，而且训练端空闲，就可以**取出 buffer 对应的数据开始训练**。

例如，取出 Buffer $$v_0$$ 的数据后，训练端开始 $$v_0\rightarrow v_1$$ 的更新。只有更新完成并发布 $$v_1$$，新的 trajectory 才能使用这版权重生成。在 $$\eta=1$$ 时，它们可以预留的位置也随之变为 Buffer $$v_1$$、Buffer $$v_2$$。

反过来，后面的 buffer 提前 *READY*，也不能抢先训练。Buffer $$v_2$$ 必须等 $$v_1\rightarrow v_2$$ 的更新完成，轮到自己时再使用。

最终的效果就是：**batch 攒齐决定训练何时能开始，训练完成产生的新版本则让后续 trajectory 获得更靠后的合法位置去 *Reserve*。** 它们相互衔接，使得生成和训练在此期间持续重叠运行。

从训练端看，*READY* 只说明某个 batch 的槽位已经填满，并不意味着它可以跳过前面的训练轮次。训练端仍按 buffer 版本顺序取用数据，生成端则在训练进行期间继续推进。只有当一轮更新完成并发布新版本，新的 trajectory 才能获得更靠后的合法范围。也正因为这个反馈存在，准入数量不能只看 rollout 端当前有多少空闲并发名额。

## Part II：演示示例

### 5. 端到端流程

下面使用 $$\eta=1$$、**batch size 为 $$4$$** 的设置作为一个代表性的示例。另外，我们为 rollout 侧设置 **max concurrency 为 $$6$$**，表示最多同时生成 $$6$$ 条 trajectory。这是可调的运行参数：可以根据 rollout instance 数量、每个 instance 的并发能力，以及 KV cache 容量、序列长度等选择。它不改变前面的 staleness 控制协议。

值得一提的是，$$6 < (\eta+1)\times \text{batch size} = 8$$。$$6$$ 个并发名额限制“同时跑多少条”，$$8$$ 个合法槽位则意味着全部 *Reserved*、*Occupied* 和 *Empty* 加起来不会超过 $$8$$。在这里，每个预留对应一条正在生成的 trajectory，所以 *Reserved* 数就是正在生成的数量。如果所有合法槽位都满了，即使生成并发数尚未达到上限，也要停止准入。

下面，我们来看一下完整的端到端演示，一共跑了 $$3$$ 个批次：

<iframe class="protocol-player" src="{{ '/assets/blog/psrl-staleness/animation/index.html?embed=1&v=20261031-clock' | relative_url }}" title="Reserve-Occupy 协议可交互动画" loading="lazy" width="100%" height="800" frameborder="0" allowfullscreen></iframe>


左侧每行是一条 trajectory，进度条总长度与这条 trajectory 从准入到完成的用时成比例。$$v_{\text{traj}}$$ 为准入时赋予 trajectory 的版本编号。右侧是 $$3$$ 个用于训练的 staleness buffer。演示动画中的移动是为了把因果关系画清楚，协议本身的 *Reserve* 和 *Occupy* 都是瞬间完成的。

### 6. 关键步骤

这段演示最终得到 $$3$$ 个训练批次：Buffer $$v_0$$ 的 `[A,D,C,E]`、Buffer $$v_1$$ 的 `[F,G,H,B]`、Buffer $$v_2$$ 的 `[J,K,L,I]`，并依次进行了训练。

下面我们选取演示动画中几个关键的节点展开分析（如果你已经通过动画理解了这套协议，可以直接跳过这一部分）。

<details markdown="1" class="protocol-detail" id="step-1">
<summary>6.1. 一开始，同时准入 A–F</summary>

初始权重为 $$v_0$$，$$\eta=1$$，batch size 为 $$4$$，max concurrency 为 $$6$$。合法范围内共有 $$8$$ 个槽位，先准入 A–F，并在准入时为它们写入 $$v_{\text{traj}}=0$$。G–L 此时还没有版本，也还没有开始生成。

![关键片段：初始一次准入 A–F]({{ '/assets/blog/psrl-staleness/clips/01-initial-admission.gif' | relative_url }})

*Reserve* 按“合法范围内尽量靠后”的规则依次落位，A–D 先进入 Buffer $$v_1$$，E、F 再进入 Buffer $$v_0$$。$$6$$ 个并发名额已经用完，因此 G 暂时不能开始。

</details>

<details markdown="1" class="protocol-detail" id="step-2">
<summary>6.2. A 完成并 <em>Occupy</em>，马上准入 G</summary>

A 完成后先离开原来的 *Reserved*，F 补到释放的槽位，A 再放入当前最早的合法 *Empty*。A 变成 *Occupied* 后立即释放一个 rollout 并发名额，Buffer $$v_0$$ 仍有合法空槽，所以同一时刻准入 G，并发恢复到 $$6$$。

![关键片段：A Occupy 后立即补入 G]({{ '/assets/blog/psrl-staleness/clips/02-first-occupy.gif' | relative_url }})

</details>

<details markdown="1" class="protocol-detail" id="step-3">
<summary>6.3. D 完成后再补 H，随后停止准入</summary>

D 完成后，G 的 *Reserved* 后移到释放的位置，D *Occupy* 到 Buffer $$v_0$$，H 随即准入。之后 C 完成，但合法范围内的 Buffer $$v_0$$ 和 Buffer $$v_1$$ 已占满 $$8$$ 个槽位，所以即使并发降到 $$5$$，I 也不能开始。Buffer $$v_2$$ 对当前 $$v_0$$ 版本仍然太晚。

![关键片段：补入 H 后，合法 buffer 容量耗尽]({{ '/assets/blog/psrl-staleness/clips/05-occupy-then-admit.gif' | relative_url }})

</details>

<details markdown="1" class="protocol-detail" id="step-4">
<summary>6.4. 第 1 批攒满，训练和生成并行</summary>

A、D、C、E 变成 *Occupied* 后，Buffer $$v_0$$ 变成 *READY*，训练端取走这 $$4$$ 条并开始 $$v_0\rightarrow v_1$$ 的更新。训练进行时，B、F、G、H 仍在生成，F、G 可以继续 *Occupy* 到 Buffer $$v_1$$。

![关键片段：首批攒满并开始训练]({{ '/assets/blog/psrl-staleness/clips/03-first-consume-training.gif' | relative_url }})

更新结束前，Buffer $$v_1$$ 已没有可用槽位，Buffer $$v_2$$ 对 $$v_0$$ trajectory 也不合法，因此，为了保证 staleness 不超限，不会因为并发暂时下降就提前准入 I。

</details>

<details markdown="1" class="protocol-detail" id="step-5">
<summary>6.5. 第 1 轮训练完成，补入 I–L</summary>

$$v_0\rightarrow v_1$$ 完成后，新的 trajectory 可以使用 $$v_1$$，合法范围扩展到 Buffer $$v_2$$。H、B 仍在生成，刚好空出 $$4$$ 个并发名额，于是 I–L 同时准入，并获得 $$v_{\text{traj}}=1$$。

![关键片段：H、B 接近完成，发布 $$v_1$$ 后准入 I–L]({{ '/assets/blog/psrl-staleness/clips/04-update-unlocks-buffer.gif' | relative_url }})

</details>

<details markdown="1" class="protocol-detail" id="step-6">
<summary>6.6. 旧 trajectory 不能后移，J 从下一批首个槽位开始</summary>

J 完成时，Buffer $$v_1$$ 已没有空槽，H、B 的 *Reserved* 也不能后移到 $$v_2$$，因为版本差会变成 $$2$$。所以 J 只能在 Buffer $$v_2$$ 内整理，并 *Occupy* 当前最早的合法空槽，I 的预留顺势后移。

![关键片段：J 无法挤进前一批，写入 Buffer $$v_2$$ 第 $$1$$ 个槽位]({{ '/assets/blog/psrl-staleness/clips/07-blocked-next-buffer.gif' | relative_url }})

</details>

<details markdown="1" class="protocol-detail" id="step-7">
<summary>6.7. B 和 I 两条长尾，先后补齐两个 batch</summary>

H 完成后，Buffer $$v_1$$ 只差 B，Buffer $$v_2$$ 仍在等待 I。K 先完成并写入 $$v_2$$，B 在 K 之后、L 之前完成。B 的 *Occupy* 让 Buffer $$v_1$$ 变成 *READY*，训练端开始 $$v_1\rightarrow v_2$$，而 L 和 I 继续与训练并行生成。

![关键片段：K 先完成，B 补齐开训，L 随后完成]({{ '/assets/blog/psrl-staleness/clips/06-b-long-tail.gif' | relative_url }})

I 最后完成后，Buffer $$v_2$$ 也变成 *READY*，等上一轮更新结束再开始下一批训练。可见，训练端在处理上一批时，下一批可以先达到 *READY* 状态。等更新结束后，已经准备好的 batch 会立即接上。

</details>

### 7. 扩展协议

上面的演示展示了最简单情形下的 staleness 控制协议。为了满足实际算法和系统优化的需求，我们还可以基于这套协议做以下扩展。

#### 7.1. Partial Rollout

一条 trajectory 可以由多个版本逐步生成（即 partial rollout，详见系列博客第一篇），而这些版本一定是单调递增的。因此，我们在准入时记录的 $$v_{\text{traj}}$$ 就是整条 trajectory 使用的最早模型版本，而后续片段的版本只会逐步增大。比如前缀来自 $$v_0$$，后缀来自 $$v_1$$，整条 trajectory 的标签仍然按 $$v_0$$ 计算。最终放置时仍按这个最早版本检查

$$
\begin{equation*}
v_{\text{buffer}}\leq v_{\text{traj}}+\eta
\end{equation*}
$$

因此，partial rollout 完全不影响协议的 staleness 约束检查：尽管 partial rollout 改变了具体的生成过程，但不会改变准入时的 $$v_{\text{traj}}$$，也因此不会改变最终的放置判定。故而，对于 partial rollout，我们无需放宽版本上限或另设规则，仍可使用同一套约束检查。

#### 7.2. *Reserved* 槽位丢弃

尽管我们设计这套协议的初衷是在不丢弃任何数据的情况下保证 staleness，但对于耗时过长或检测出问题的 trajectory，我们也支持追踪 *Reserved* 槽位的状态，并主动丢弃对应的 trajectory。这样一方面可以进一步提高系统性能，另一方面也能让协议更加健壮，不会因为某个 trajectory 出现错误或卡死而导致整个协议瘫痪。下面，我们用一个超时错误作为例子：当训练端开始等待 buffer 攒齐时，我们检查最近的 buffer；如果它处于 *STUCK* 状态，就可以开始计时。当等待时间超过阈值后，就清空该 buffer 中剩余的所有 *Reserved* 槽位。此时，如果后面的 buffer 中已经有 *Occupied* 的数据，可以直接将其前移，填补空位，尽快攒满 buffer。另一方面，由于空出了槽位，我们可以从数据集中准入新的数据进行生成，来弥补丢弃的数据。

![Reserved Timeout：B、C 超时退出，D 前移补位，再准入 G、H 并建立新的 Reserved]({{ '/assets/blog/psrl-staleness/extensions/stuck-timeout-recovery.gif' | relative_url }})

#### 7.3. *Occupied* 槽位筛选

此外，另一种主动过滤也可能发生在条目已经变成 *Occupied* 之后。例如，在后处理中，如果我们确认某条数据无效（不提供任何梯度，或数据本身存在问题、属于脏数据等），我们同样支持清空这个槽位，再从后方取出 *Occupied* 数据填入缺口，并继续准入新的数据。下面的例子中，C 完成后先写入 $$v_0$$，随后被过滤，再将 $$v_1$$ 中已处于 *Occupied* 状态的 D 前移补位。尚未完成的 E、F 仍保留为 *Reserved*。随后准入 G，并为它预留空出的槽位。

![Occupied Filtering：C 完成后被过滤，D 前移补位，再准入 G 并建立新的 Reserved]({{ '/assets/blog/psrl-staleness/extensions/occupied-filter.gif' | relative_url }})

#### 7.4. GRPO Group

在上面所有例子中，我们的 *Reserve* 以及 *Occupy* 都是以 trajectory 为粒度进行的。但在实际中，GRPO 这类算法通常会为一个 prompt 生成多个 trajectory，这样的 trajectory group 中的成员必须全部完成，才可以实际用于训练。在这种情况下，我们将推理和训练中同一 prompt 的一组 trajectory 视作一个基本单位，然后让 staleness buffer 中的一个槽位对应一整个 trajectory group：*Reserve* 为整个 group 留位置，而 *Occupy* 则需要等待全组成员和后处理都就绪后再执行。


![GRPO group：成员完成后删除 Reserved，再 Occupy 最早合法槽位]({{ '/assets/blog/psrl-staleness/extensions/grpo-group.gif' | relative_url }})

#### 7.5. Group-level 过采样

为了减少一个 group 内部的 trajectory 长尾带来的等待，可以为一个 group 同时生成超过所需数量（group size）的候选。只要凑齐 group size，就立刻提交已选成员，剩余候选直接丢弃，不再等待长尾候选生成完成。

![Group-level redundancy：达到所需成员数后停止等待多余候选]({{ '/assets/blog/psrl-staleness/extensions/redundant-group.gif' | relative_url }})


#### 7.6. Batch-level 过采样

此外，我们也可以生成数量超过所需 batch 大小（batch size）的 trajectory group，这样可以筛掉整体相对长尾的 group。在这种情况下，buffer 的槽位数会多于实际 batch 的大小，而当这些额外槽位生成的数据不可能参与到实际 batch 的组建时，我们再将其丢弃。下面举一个具体的例子：batch 大小为 $$2$$，而 buffer 的槽位数为 $$3$$。此时，每个 buffer 攒够 $$2$$ 个 *Occupied* 就可以变成 *READY* 并组建 batch 进行训练，而分界线右侧的第 $$3$$ 个槽位则表示冗余的容量。开始时，$$v_0$$ 有 $$1$$ 个 *Occupied* 和 $$2$$ 个 *Reserved*，$$v_1$$ 有 $$2$$ 个 *Occupied* 和 $$1$$ 个 *Reserved*。最先完成的候选放入 $$v_0$$ 后，$$2$$ 个 batch 都达到目标数量，协议会判断剩余 $$2$$ 个 *Reserved* 产生的数据不可能再有合法的放置位置，因此会一起丢弃，不再参与训练（协议只需要在每次有 buffer 变为 *READY* 时进行一次判断即可）。


![Batch-level redundancy：G3 完成并填入 $$v_0$$，两个 batch 均 READY，G1 和 G2 被 Abort]({{ '/assets/blog/psrl-staleness/extensions/redundant-batch.gif' | relative_url }})


## Part III：实验效果

在前面的两部分中，我们介绍了这套 *Reserve*-*Occupy* 协议的设计，并通过动画展示了它的具体执行过程。接下来，我们通过实验验证这套协议是否能严格控制 staleness，并进一步比较它与其他控制机制的吞吐表现。

与上一篇博客采用的研究方法一致，我们基于自研的 [**SimRL**](https://github.com/psrl-project/SimRL) 模拟器开展实验，在相同的硬件、模型和负载设置下对比各套机制。实验分为两部分：首先检查每个训练 batch 的实际 staleness 是否满足上限，再比较各套机制的有效训练吞吐。

### 8. 实验设置

我们模拟 Qwen3-8B 在一台配有 $$8$$ 张 H20 的机器上进行 RL 训练。其中，$$2$$ 张卡用于推理，对应 $$2$$ 个 rollout instance，各自采用 $$\mathrm{TP}=1$$。另外 $$6$$ 张卡用于训练，使用 Megatron，设置为 $$\mathrm{TP}=2$$、$$\mathrm{PP}=3$$、$$\mathrm{DP}=1$$。我们采用 GRPO 算法，group size 为 $$8$$，每个训练 batch 包含 $$16$$ 个 prompt，也就是 $$16\times8=128$$ 条 trajectory，并默认开启 partial rollout。

我们使用来自 dapo-math-17k 真实训练过程的负载 trace，包含 $$1024$$ 个 prompt group、共 $$8192$$ 条 response。原始负载的 prompt 长度上限设置为 $$1024$$ token，response 长度上限设置为 $$38912$$ token，具体分布如下。

[![原始负载的 prompt 与 response 长度分布]({{ '/assets/blog/psrl-staleness/exp/length_histograms-blog.png' | relative_url }})]({{ '/assets/blog/psrl-staleness/exp/length_histograms-blog.png' | relative_url }})

图中的两条虚线分别标出了均值和 p95，其中 p95 表示第 $$95$$ 百分位数。这批负载的 prompt 长度中位数为 $$138$$、p95 为 $$246$$，而 response 长度中位数为 $$8673$$、p95 超过 $$25000$$。可以看到，prompt 普遍较短，而 response 的长尾十分明显。一条 trajectory 生成得越久，它的生成版本与后续训练版本之间的差距就可能越大，因此，这类负载尤其需要严格的 staleness 控制。

为了进一步观察长尾的影响，我们还在原始 trace 上构造了一组更极端的负载：选取 $$3$$ 个 group，让每组全部 $$8$$ 条 response 一起变长，组内平均长度增加到约 $$45000$$ token，约为原始 trace 中最长一组平均长度 $$34449$$ 的 $$1.3$$ 倍。这样，长尾不仅出现在单条 trajectory 上，也会拖慢整个 group 的完成进度。

对于构造后的负载，我们将 response 上限放宽到 $$131072$$ token，不再按原始的 $$38912$$ 截断。下图展示了调整后的长度分布，并在右上角放大了长尾部分，方便观察这些数量较少、长度却明显更大的 response。

[![构造长尾负载的长度分布，右上角放大超长 response 所在区间]({{ '/assets/blog/psrl-staleness/exp/length_histograms_longtail-blog.png' | relative_url }})]({{ '/assets/blog/psrl-staleness/exp/length_histograms_longtail-blog.png' | relative_url }})

### 9. 正确性验证：协议是否全程不超过 staleness 上限

我们首先验证协议的正确性：给定 staleness 上限 $$\eta$$，这套协议能否确保每个训练 batch 中的数据都不超过这一上限？

我们固定每个 rollout instance 的 max concurrency 为 $$40$$。分别设置 $$\eta\in\{0,1,2,3,4,5\}$$，在两种负载下各运行 $$100$$ 个训练 batch。

下图展示了原始负载的实验结果。每个子图对应一个 $$\eta$$，每根柱子对应一个训练 batch，不同深浅的颜色表示这一批中不同 staleness 的 trajectory 数量。

[![原始负载下，六种 staleness 上限对应的逐 batch 构成]({{ '/assets/blog/psrl-staleness/exp/staleness_composition_grid-blog.png' | relative_url }})]({{ '/assets/blog/psrl-staleness/exp/staleness_composition_grid-blog.png' | relative_url }})

我们再用加入更长尾 group 的负载进行同样的实验，结果如下。

[![构造长尾负载下，六种 staleness 上限对应的逐 batch 构成]({{ '/assets/blog/psrl-staleness/exp/staleness_composition_grid_longtail-blog.png' | relative_url }})]({{ '/assets/blog/psrl-staleness/exp/staleness_composition_grid_longtail-blog.png' | relative_url }})

从这两张图中，我们可以看到，每根柱子的总高度始终为 $$128$$，说明每个训练 batch 都包含完整的 $$128$$ 条 trajectory。与此同时，各子图中实际出现的 staleness 都没有超过对应的 $$\eta$$。两种负载、$$6$$ 种上限、每组 $$100$$ 个 batch，合计展示的 $$1200$$ 个 batch 中都没有出现超限。

值得一提的是，当 $$\eta=5$$ 时，两种负载下观测到的最大 staleness 都只有 $$4$$。这是因为 **$$\eta$$ 限制的是数据参与训练时允许的最大版本差，并不要求数据一定等到这个上限才参加训练。** 只要当前有更早的合法位置，我们的协议就会通过 *Occupy* 尽早安排这些数据。

这些实验结果与前面对协议的分析一致：即使加入更极端的长尾，我们的协议在这些测试中仍然能够严格控制 staleness。

### 10. 性能验证：相同 max staleness 下的吞吐对比

接下来，我们把这套协议与前文分析的另外 $$3$$ 种具有严格 staleness 控制的机制放在一起比较。（1）NeMo-RL 的 `in_order`：在准入时绑定训练轮次，确保 staleness 严格受控。（2）verl 的 `drop` 以及 TRL：在 L1 限制准入、在 L3 过滤超限结果。（3）ROLL：进一步在 L2 提前淘汰超限的生成任务。

本组实验覆盖 $$\eta\in\{1,2,3\}$$，每次模拟 $$10$$ 个训练 batch。我们对齐各机制所允许的最大 staleness $$\eta$$。特别地，verl 的 L1 准入额度还可以单独调节，这里固定 `num_warmup_batches` 为默认值 $$1$$，不随 $$\eta$$ 一起增大。

我们使用**有效吞吐**作为比较指标，即实际被 RL 训练采用的 token 总数，除以该次运行的总时长，单位为 tokens/s。因此，即使某套机制生成了更多数据，其中没有参与训练的部分也不会计入有效吞吐。

在下面两张图中，我们用 Ours 表示我们这套 *Reserve*-*Occupy* 协议。**柱子的实色部分和顶部数值表示有效吞吐。** 上方两种阴影分别表示在 L2、L3 被丢弃的部分，这些部分不计入有效吞吐。因此，比较性能时应看实色部分，不能把包含阴影的总柱高当作有效吞吐。绿色倍率表示我们的协议相对于对应基线的吞吐加速比。

原始负载下的结果如下：

[![原始负载下的有效吞吐与 L2、L3 丢弃开销]({{ '/assets/blog/psrl-staleness/exp/ours_vs_nemo_vs_verl_vs_roll_throughput_v10-blog.png' | relative_url }})]({{ '/assets/blog/psrl-staleness/exp/ours_vs_nemo_vs_verl_vs_roll_throughput_v10-blog.png' | relative_url }})

长尾更显著时的结果如下：

[![构造长尾负载下的有效吞吐与 L2、L3 丢弃开销]({{ '/assets/blog/psrl-staleness/exp/ours_vs_nemo_vs_verl_vs_roll_throughput_v10_longtail-blog.png' | relative_url }})]({{ '/assets/blog/psrl-staleness/exp/ours_vs_nemo_vs_verl_vs_roll_throughput_v10_longtail-blog.png' | relative_url }})

**首先，我们对比同样不依赖超限丢弃的 NeMo-RL `in_order`。** 在原始负载下，我们的有效吞吐约为它的 $$1.05\text{–}1.06$$ 倍，长尾负载下则约为 $$1.08\text{–}1.13$$ 倍。

这一差距与前文讨论的两套机制的区别相吻合。`in_order` 在生成前就固定了训练批次，而我们的协议通过 *Reserve* 为每个 group 保留合法位置，再通过 *Occupy* 根据实际完成顺序调整放置。这样，完成较快的 group 可以更早凑齐当前 batch，长尾 group 则能在合法范围内后移。因此，相比于固定批次安排，我们的协议更加灵活，在维持 staleness 约束且不丢弃数据的前提下，可以最大程度地优化系统性能。

**其次，我们对比依赖 L3 过滤的 verl `drop` 和 TRL。** 原始负载、$$\eta=1$$ 时，verl 有明显的超限丢弃，我们的有效吞吐约为它的 $$1.19$$ 倍。当 $$\eta$$ 增大到 $$2$$ 或 $$3$$ 后，verl 不再出现超限丢弃，吞吐差距也缩小到约 $$1.03$$ 倍。这是因为，在这组负载下，较紧的 L1 准入（`num_warmup_batches=1`）配合更宽的 staleness 上限，就已经能够彻底避免 staleness 超限导致的浪费了。

不过，当我们进一步加重负载的长尾效应时，即使 $$\eta=2$$ 或 $$3$$，verl 仍有数据在 L3 被丢弃。此时，我们的有效吞吐分别约为它的 $$1.13$$、$$1.12$$ 倍。可见，L1 的数量限制能够减少积压，但遇到更慢的 group，仍可能需要在 L3 丢弃数据来保证 staleness 上限。

而对于 TRL，L1 准入量随 staleness 设置一同扩大，并未被限制为固定值，这导致在所有的测试点上，TRL 都出现了 L3 丢弃。不过，也不能仅凭丢弃量就判定它一定更慢。例如，原始负载、$$\eta=1$$ 时，TRL 的有效吞吐约为 $$1584$$ tokens/s，高于 verl 的 $$1511$$。这也说明，限制准入是在生成并发与丢弃浪费之间做取舍，实际效果并不能简单预测，需要看最终的有效吞吐。

**最后，我们对比进一步加入 L2 淘汰的 ROLL。** ROLL 可以提前停止已经超限的 trajectory，避免继续生成后续 token，但具体收益还取决于停止时机和之后的调度。例如，长尾负载、$$\eta=1$$ 时，ROLL 的有效吞吐约为 $$1421$$ tokens/s，高于 verl 的 $$1282$$，却低于 TRL 的 $$1471$$。原始负载、$$\eta=2$$ 时，ROLL 的 $$1520$$ 也低于另外两者。因此，L2 提前淘汰有助于减少继续生成的浪费，但也不能保证整体吞吐一定更高。

总的来看，在本次两种负载、$$3$$ 个上限的测试点中，我们这套 *Reserve*-*Occupy* 协议的有效吞吐均为最高，同时没有依靠丢弃超限数据来维持 staleness 上限。我们在准入时为每条 trajectory 保留合法位置，在完成后再根据当前状态调整放置，既避免了超限丢弃造成的浪费，也为更早凑齐训练 batch 留出了空间。

这里，我们比较的是 [**SimRL**](https://github.com/psrl-project/SimRL) 上模拟出的各套机制的表现，真实 GPU 上的端到端吞吐和训练收敛仍需要进一步验证。而对于本文关心的问题，这组实验已经展示了我们这套协议的效果：**在所测试的负载下，我们通过 L1 准入与 L3 放置的配合，既严格控制了 staleness，也取得了更高的有效训练吞吐。**

## Part IV：小结

我们在这篇博客中介绍了 *Reserve*-*Occupy* 协议的设计，其核心是在生成准入时，也把未来训练位置的安排纳入考虑：**用 *Reserve* 先留一个合法槽位，*Occupy* 根据完成顺序调整放置，训练端再按版本取走完整 batch。**

这样，L1 能看到训练侧还剩哪些合法位置，L3 调整位置时也会保留未完成 trajectory 的去向。完成较快的 trajectory 不必始终受最初 batch 安排的约束，完成较慢的 trajectory 也不会因为一直被延后，最终被安排到不合法的位置参与训练，从而违反 staleness 约束。严格的 staleness 控制与运行时灵活性，正是通过这套由 L1 与 L3 共同维护的、基于 buffer 的 *Reserve*-*Occupy* 协议得到保证的。

目前，这套协议已经完整实现在我们的 RL 框架 [**PSRL**](https://github.com/psrl-project/psrl) 中。欢迎试用我们的框架，也欢迎在 issue 中提出与博客或框架相关的问题和建议。如果这篇博客对你有帮助，希望你能顺手给 PSRL 点个 star⭐，你的支持也是我们持续完善框架、分享技术博客的动力！
