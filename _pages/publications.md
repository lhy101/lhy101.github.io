---
layout: archive
title: "Publications"
permalink: /publications/
author_profile: true
---

{% if author.googlescholar %}
  You can also find my articles on <u><a href="{{author.googlescholar}}">my Google Scholar profile</a>.</u>
{% endif %}

{% include base_path %}

{% for post in site.publications reversed %}
  {% include archive-single.html %}
{% endfor %}

\* indicates equal contribution

- **[Preprint, ChinaSys 2025 Oral]** <u><a href="https://arxiv.org/abs/2410.13333">Hetu v2: A General and Scalable Deep Learning System with Hierarchical and Heterogeneous Single Program Multiple Data Annotations</a></u> 
  *<u>Haoyang Li</u>, Fangcheng Fu, Hao Ge, Sheng Lin, Xuanyu Wang, Jiawen Niu, Xupeng Miao, Bin Cui*

- **[Preprint]** <u><a href="https://arxiv.org/abs/2412.07894">Demystifying Workload Imbalances in Large Transformer Model Training over Variable-length Sequences</a></u> 
  *<u>Haoyang Li</u>, Fangcheng Fu, Sheng Lin, Hao Ge, Xuanyu Wang, Jiawen Niu, Jie Jiang, Bin Cui*

- **[Preprint]** <u><a href="https://arxiv.org/abs/2412.07894">Ampelos: Efficient Checkpointing and Fault-Recovery for Elastic Distributed Training</a></u> 
  *Xuanyu Wang, Fangcheng Fu, <u>Haoyang Li</u>, Hao Ge, Sheng Lin, Jiawen Niu, Bin Cui*

---

- **[SIGMOD 2025]** <u><a href="https://arxiv.org/abs/2410.13333">Malleus: Straggler-Resilient Hybrid Parallel Training of Large-scale Models via Malleable Data and Model Parallelization</a></u> 
  *<u>Haoyang Li</u>\*, Fangcheng Fu\*, Hao Ge, Sheng Lin, Xuanyu Wang, Jiawen Niu, Yujie Wang, Hailin Zhang, Xiaonan Nie, Bin Cui*


- **[VLDB 2025]** <u><a href="TBD">LobRA: Multi-tenant Fine-tuning over Heterogeneous Data</a></u>
  *Sheng Lin\*, Fangcheng Fu\*, <u>Haoyang Li</u>, Hao Ge, Xuanyu Wang, Jiawen Niu, Yaofeng Tu, Bin Cui*

- **[SOSP 2024]** <u><a href="https://dl.acm.org/doi/10.1145/3694715.3695969">Enabling Parallelism Hot Switching for Efficient Training of Large Language Models</a></u>
  *Hao Ge\*, Fangcheng Fu\*, <u>Haoyang Li</u>, Xuanyu Wang, Sheng Lin, Yujie Wang, Xiaonan Nie, Hailin Zhang, Xupeng Miao, Bin Cui*

- **[AAAI 2024, MLSys Workshop NeurIPS 2023]** <u><a href="https://ojs.aaai.org/index.php/AAAI/article/view/29599">Accelerating text-to-image editing via cache-enabled sparse diffusion inference</a></u>
  *Zihao Yu, <u>Haoyang Li</u>, Fangcheng Fu, Xupeng Miao, Bin Cui*

