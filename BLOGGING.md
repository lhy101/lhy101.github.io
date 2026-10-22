# 添加博客

文章放在 `_blog/`，列表会按日期倒序自动更新，无需修改导航或列表模板。

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
- 折叠内容使用 `<details markdown="1"><summary>标题</summary>`，内部 Markdown 前后留空行，以便 Kramdown 正确解析。需要加入目录时，添加 `class="framework-detail"`，其标题会作为四级目录项，点击后自动展开内容。
- 独立公式使用单独一行的 `$$` 包裹，公式块前后各留一空行，避免被解析成行内公式。不需要编号时，在公式块内使用 `\begin{equation*}` 和 `\end{equation*}`。
- `lang` 可改为 `en`；`series` 可省略。

## 第一篇文章的来源

`_blog/async-rl-staleness.md` 来自博客工作目录中的
`async-rl-staleness-blog.zh-CN.md`。正文只调整了图片路径和折叠区的
Markdown 解析属性，并将独立公式设为不编号的居中公式。三张正文 PNG 保留原图；列表封面为框架对比图的独立裁切版。

如需同步原文章，运行（需要 Python 和 Pillow）：

```powershell
python scripts/import-staleness-blog.py 'C:\Users\lihaoyang\Desktop\workspace\blog'
```

导入会更新该文章及其配图、封面。其他博客不受影响。确认源文章与配图后再运行。

## 本地构建

有 Ruby / Bundler 环境时，使用站点现有的 GitHub Pages 依赖：

```powershell
bundle install
bundle exec jekyll serve
```

打开 `http://127.0.0.1:4000/` 和 `/blog/` 检查桌面、手机排版以及折叠区、公式、配图。
