"""Import the article and its three figures, preserving the source prose.

Usage: python scripts/import-staleness-blog.py /path/to/blog
Requires Pillow only for the separate list cover crop.
"""
import argparse
from pathlib import Path
import re
import shutil

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory containing the original blog")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.source / "async-rl-staleness-blog.zh-CN.md"
    original = source.read_text(encoding="utf-8")
    title = original.splitlines()[0].removeprefix("# ")
    asset_dir = root / "assets/blog/async-rl-staleness"
    asset_dir.mkdir(parents=True, exist_ok=True)
    figures = re.findall(r"!\[[^\]]*\]\((assets/[^)]+)\)", original)
    for figure in figures:
        shutil.copy2(args.source / figure, asset_dir / Path(figure).name)

    # Use the full comparison table. Remove its numbered heading and footer.
    with Image.open(asset_dir / "systems-layer-map.png") as image:
        image.crop((48, 216, 1232, 1061)).save(asset_dir / "cover.webp", quality=94)

    # Adapt paths and Markdown formatting while preserving the source prose.
    article = re.sub(
        r"\]\(assets/async-rl-staleness-blog/([^)]+)\)",
        lambda match: "]({{ '/assets/blog/async-rl-staleness/" + match[1] + "' | relative_url }})",
        original,
    )
    article = article.replace('<details class="framework-detail"', '<details markdown="1" class="framework-detail"')
    # Kramdown requires blank lines around display math to avoid inline output.
    article = article.replace(
        r"\text{token staleness}=V_{\text{train}}-V_{\text{gen}}",
        r"\text{token staleness} = V_{\text{train}} - V_{\text{gen}}",
    )
    article = article.replace(
        r"\text{token staleness} = V_{\text{train}} - V_{\text{gen}}",
        r"\begin{equation*}" + "\n" +
        r"\text{token staleness} = V_{\text{train}} - V_{\text{gen}}" + "\n" +
        r"\end{equation*}",
    )
    article = re.sub(
        r"\n*\n\$\$\n([\s\S]*?)\n\$\$\n*",
        lambda match: "\n\n$$\n" + match[1].strip() + "\n$$\n\n",
        article,
    )
    metadata = f'''---
layout: blog-post
title: "{title}"
description: "从 trajectory 的生命周期出发，对照 10 个异步 RL 框架的源码，分析不同框架的三层 staleness 控制机制。"
date: 2026-10-22
lang: zh-CN
series: "PSRL / 01"
permalink: /blog/async-rl-staleness/
cover: /assets/blog/async-rl-staleness/cover.webp
cover_alt: "10 个异步 RL 框架的 L1、L2、L3 控制机制与严格 staleness 控制对比"
cover_width: 1184
cover_height: 845
author_profile: false
---

'''
    (root / "_blog").mkdir(exist_ok=True)
    (root / "_blog/async-rl-staleness.md").write_text(metadata + article, encoding="utf-8")
    print(f"Imported {title}: {len(figures)} figures and one cropped cover.")


if __name__ == "__main__":
    main()
