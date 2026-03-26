---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <b><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</b>
{% endif %}

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}

\* indicates equal contribution

- **[Preprint]** <a href="https://arxiv.org/abs/2601.12784">Unleashing Efficient Asynchronous RL Post-Training via Staleness-Constrained Rollout Coordination</a> 
  *<b>Haoyang Li</b>\*, Sheng Lin\*, Fangcheng Fu, Yuming Zhou, Xiaodong Ji, Yanfeng Zhao, Lefeng Wang, Jie Jiang, Bin Cui*

---

- **[OSDI'26, ChinaSys'25 Oral]** <a href="https://arxiv.org/abs/2504.20490">Hetu v2: A General and Scalable Deep Learning System with Hierarchical and Heterogeneous Single Program Multiple Data Annotations</a> 
  *<b>Haoyang Li</b>, Fangcheng Fu, Hao Ge, Sheng Lin, Xuanyu Wang, Jiawen Niu, Xupeng Miao, Bin Cui*

---

- **[PPoPP'26]** <a href="https://dl.acm.org/doi/abs/10.1145/3774934.3786445">Elastor: Elastic and Efficient Model Partitioning and Checkpointing for Fault-tolerant Distributed DL Training</a> 
  *Xuanyu Wang, Fangcheng Fu, <b>Haoyang Li</b>, Hao Ge, Sheng Lin, Jiawen Niu, Bin Cui*

---

- **[SIGMOD'26]** <a href="https://dl.acm.org/doi/10.1145/3769802">Hydraulis: Balancing Large Transformer Model Training via Co-designing Parallel Strategies and Data Assignment</a>
  *<b>Haoyang Li</b>, Fangcheng Fu, Sheng Lin, Hao Ge, Xuanyu Wang, Jiawen Niu, Jinbao Xue, Yangyu Tao, Di Wang, Jie Jiang, Bin Cui*

---

- **[SIGMOD'25]** <a href="https://dl.acm.org/doi/10.1145/3725322">Malleus: Straggler-Resilient Hybrid Parallel Training of Large-scale Models via Malleable Data and Model Parallelization</a>
  *<b>Haoyang Li</b>\*, Fangcheng Fu\*, Hao Ge, Sheng Lin, Xuanyu Wang, Jiawen Niu, Yujie Wang, Hailin Zhang, Xiaonan Nie, Bin Cui*

---

- **[VLDB'25]** <a href="https://dl.acm.org/doi/10.14778/3742728.3742752">LobRA: Multi-tenant Fine-tuning over Heterogeneous Data</a>
  *Sheng Lin\*, Fangcheng Fu\*, <b>Haoyang Li</b>, Hao Ge, Xuanyu Wang, Jiawen Niu, Yaofeng Tu, Bin Cui*

---

- **[SOSP'24]** <a href="https://dl.acm.org/doi/10.1145/3694715.3695969">Enabling Parallelism Hot Switching for Efficient Training of Large Language Models</a>
  *Hao Ge\*, Fangcheng Fu\*, <b>Haoyang Li</b>, Xuanyu Wang, Sheng Lin, Yujie Wang, Xiaonan Nie, Hailin Zhang, Xupeng Miao, Bin Cui*

---

- **[AAAI'24, MLSys Workshop NeurIPS'23]** <a href="https://ojs.aaai.org/index.php/AAAI/article/view/29599">Accelerating text-to-image editing via cache-enabled sparse diffusion inference</a>
  *Zihao Yu, <b>Haoyang Li</b>, Fangcheng Fu, Xupeng Miao, Bin Cui*

