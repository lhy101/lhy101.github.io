# 添加博客

文章放在 `_blog/`，列表按可选的 `list_order` 从大到小排列，同序号文章按日期倒序显示。staleness 系列固定按（三）、（二）、（一）排列，日期保留原值。

文章底部自动显示上一篇、下一篇，阅读顺序按 `list_order` 从小到大，同序号按日期从早到晚。staleness 系列按（一）→（二）→（三）衔接，首尾篇只显示可用方向。桌面端左右排列，手机端上下排列，下方保留返回列表和回到顶部。

新建 `_blog/your-slug.md`：

```markdown
---
layout: blog-post
title: "文章标题"
description: "列表中显示的一两句摘要。"
date: 2026-10-22
lang: zh-CN
cover: /assets/blog/your-slug/cover.webp
cover_alt: "封面图片的简短说明"
cover_width: 1184
cover_height: 845
series: "可选的系列名称"
---

# 文章标题

正文……

## 第一节
```

- 文件名决定地址：`your-slug.md` 对应 `/blog/your-slug/`。如需覆盖地址，可添加 `permalink`。
- `title` 与正文的一级标题保持一致。正文使用一个一级标题，二至四级标题会自动生成分层目录。
- 配图放在 `assets/blog/your-slug/`，仅保留发布需要的文件。`cover_width`、`cover_height` 填图片实际像素大小。
- 图片示例：`![图片说明]({{ '/assets/blog/your-slug/figure.png' | relative_url }})`。文章配图可点击查看原图。
- 折叠内容使用 `<details markdown="1"><summary>标题</summary>`，内部 Markdown 前后留空行，以便 Kramdown 正确解析。框架和协议折叠区分别使用 `class="framework-detail"`、`class="protocol-detail"`；网站导入时去掉折叠标题中的小节编号，文章目录只列正文标题，折叠区仍可展开，并保留原有锚点链接。
- 独立公式使用单独一行的 `$$` 包裹，公式块前后各留一空行，避免被解析成行内公式。不需要编号时，在公式块内使用 `\begin{equation*}` 和 `\end{equation*}`。
- `lang` 可改为 `en`；`series` 可省略。
- 每篇文章标题下自动显示“全文约 ×× 字 · 预计阅读 ×× 分钟”，无需手填字数。统计包含折叠正文，中文逐字、英文和数字按词计数，排除代码块、公式和图片内容；中文按每分钟 500 字估算，可在 `_config.yml` 中用 `blog_zh_words_per_minute` 调整，其他语言沿用 `words_per_minute`。此值是阅读提示的经验设置，实际用时取决于读者对内容的熟悉程度，以及阅读图表和推导公式所需的时间。

阅读提示参考 [FixIt 文档页](https://fixit.lruihao.cn/zh-cn/documentation/content-management/introduction/)的“约 ×× 字 / 预计阅读 ×× 分钟”表述，以及 NexT 的字数与阅读时长元信息，用一行小字放在标题和目录之间。

## 阅读量统计（可选）

本站使用 GoatCounter 的公开阅读量接口，按文章 URL 统计页面访问次数。在
GoatCounter 创建站点并开启 **Allow adding visitor counts on your website** 后，
将站点的 `/count` 地址填写到 `_data/blog_counter.yml` 中的 `endpoint`：

```yaml
endpoint: "https://lhygood11229.goatcounter.com/count"
```

部署后，文章页和博客列表会显示各自的阅读量。打开文章才会记录一次访问，
博客列表只读取各篇文章的计数。新文章会自动接入，不必单独配置。

- 这里显示页面访问量，不代表独立读者数或完整读完的人数。
- 接入前的历史访问无法追溯；修改文章 URL 会影响计数归属，应保持 permalink 稳定。
- 公开计数接口可能缓存最多 4 小时，因此刷新后数字不一定立即变化。
- 留空会关闭计数；本地预览及其他域名不会请求统计服务。
- 新文章尚无数据、接口未开放、网络失败或被拦截时，计数会隐藏，不显示虚构的 0。
- 不需要在仓库里填写 API Key 或密码。统计数据及维护由配置的 GoatCounter 站点提供。

服务说明见 <https://www.goatcounter.com/help/visitor-counter>。

## 第一篇文章的来源

`_blog/async-rl-staleness.md` 来自博客工作目录中的
`async-rl-staleness-blog.zh-CN.md`。正文只调整了图片路径和折叠区的
Markdown 解析属性，并将独立公式设为不编号的居中公式。正文使用文字已转为矢量路径的 SVG；列表封面为框架对比图的高清独立裁切版。

如需同步原文章，运行（需要 Python 和 Pillow）：

```powershell
python scripts/import-staleness-blog.py 'C:\Users\lihaoyang\Desktop\workspace\blog'
```

导入会更新该文章及其配图、封面。其他博客不受影响。确认源文章与配图后再运行。

## 同步 staleness 系列三篇文章

原命令仍只导入（一）。同时更新三篇时运行：

```powershell
python scripts/import-staleness-blog.py 'C:\Users\lihaoyang\Desktop\workspace\blog' --article all
```

只导入（一）、（二）、（三）时，分别使用 `--article async-rl-staleness`、`--article simrl-staleness`、`--article psrl-staleness`。

系列的统一地址为 `/blog/async-rl-staleness-1/`、`/blog/async-rl-staleness-2/`、`/blog/async-rl-staleness-3/`。原来的三个地址通过 `redirect_from` 自动跳转，文章文件名和资源目录保持不变。

（二）保存在 `_blog/simrl-staleness.md`。列表封面来自 blog 工作区的 `throughput-cover.svg`，节选正文端到端柱状图的前 3 档 staleness 上限，保留全部 14 个配置。封面不带脚注，仅复制发布版 SVG，尺寸为 3200 × 1920。

（三）保存在 `_blog/psrl-staleness.md`，日期为 2026-10-31。导入保留原文内容、公式、折叠区和媒体结构，并迁移交互动画所需的页面与图层。动画播放器位于该文章资源目录的 `animation/index.html`，文章会按播放器发送的高度自动调整 iframe。

（三）的列表封面使用 `cover.svg`，来自 blog 工作区的 `protocol-cover.svg`，逻辑尺寸 1280 × 720。它由 `scripts/draw_psrl_cover.py` 基于正文协议图的绘图组件生成，保留原有正文图片。更新封面后再运行导入命令。

每篇文章的资源目录包含 `import-manifest.json`，记录源文 SHA-256 与复制资源的 SHA-256。导入只复制正文、封面和动画实际需要的文件，不复制实验源码、角色素材或检查截图。导入器会清理这三个文章资源目录内已不再使用的媒体和重复格式，目录中不要放置其他文件。

导入后可运行 `python scripts/check-staleness-imports.py 'C:\Users\lihaoyang\Desktop\workspace\blog'`，核对正文、公式、资源哈希和动画路径。

折叠标题中的公式会转换为 `\(...\)`，避免 MathJax 把标题拆成带编号的独立公式块。`proof-detail` 与框架折叠区共用样式，展开后会重新计算隐藏公式的布局。

播放器回归检查使用 `node scripts/check-psrl-playback.cjs`（需要 Playwright 与 Chrome，可通过 `CODEX_NODE_MODULES`、`PSRL_BROWSER_PATH` 指定）。它验证从头持续播放、画面推进、暂停、继续、拖动和重新播放，并模拟嵌入浏览器的动画回调时间起点偏移。播放器统一使用自身的 `performance.now()` 计时，避免 iframe 内出现负数进度。可用 `PSRL_PLAYER_URL` 指定线上播放器地址。

## 本地构建

有 Ruby / Bundler 环境时，使用站点现有的 GitHub Pages 依赖：

```powershell
bundle install
bundle exec jekyll serve
```

打开 `http://127.0.0.1:4000/` 和 `/blog/` 检查桌面、手机排版以及折叠区、公式、配图。
