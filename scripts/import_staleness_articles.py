"""Import staleness articles without rewriting their prose. Requires Pillow."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
POSTS = {
    'async-rl-staleness': {
        'source': 'async-rl-staleness-blog.zh-CN.md',
        'date': '2026-10-22', 'series': 'PSRL / 01', 'cover': 'cover.webp',
        'description': '我们从 trajectory 的生命周期出发，对照 10 个异步 RL 框架的源码，分析不同框架的三层 staleness 控制机制。',
        'cover_alt': '10 个异步 RL 框架的 L1、L2、L3 控制机制与严格 staleness 控制对比',
        'asset_prefix': 'assets/async-rl-staleness-blog/',
    },
    'psrl-staleness': {
        'source': 'psrl-staleness-blog.zh-CN.md',
        'date': '2026-10-31', 'series': 'PSRL / 03', 'cover': 'cover.svg',
        'description': '我们设计了一套 Reserve–Occupy 协议，在严格控制 staleness 的前提下，不丢弃任何数据，并最大化有效 RL 训练吞吐。',
        'cover_alt': 'Reserve–Occupy 协议：生成前预留合法位置，完成后尽早放置，并共同满足版本上限',
        'asset_prefix': 'assets/psrl-staleness/',
    },
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def asset_url(slug, relative):
    return "{{ '/assets/blog/" + slug + '/' + relative + "' | relative_url }}"


def adapt_article(original, slug):
    prefix = POSTS[slug]['asset_prefix']
    article = re.sub(r'\]\(' + re.escape(prefix) + r'([^)]+)\)',
                     lambda match: '](' + asset_url(slug, match[1]) + ')', original)
    article = re.sub(r'<details(?![^>]*\bmarkdown=)([^>]*)>', r'<details markdown="1"\1>', article)
    article = re.sub(r'(?m)^\$\$([^\n]+)\$\$[ \t]*$', r'$$\n\1\n$$', article)

    def display(match):
        tex = match[1].strip()
        if not tex.startswith(r'\begin{equation*}'):
            tex = '\\begin{equation*}\n' + tex + '\n\\end{equation*}'
        return '\n\n$$\n' + tex + '\n$$\n\n'

    article = re.sub(r'\n*\n\$\$\n([\s\S]*?)\n\$\$\n*', display, article)
    if slug == 'psrl-staleness':
        article = re.sub(r'src="publishing/psrl-staleness/animation-preview\.html(\?[^"]*)?"',
                         lambda match: 'src="' + asset_url(slug, 'animation/index.html' + (match[1] or '')) + '"', article)
    return article.rstrip() + '\n'


def import_animation(source, destination, copied):
    scene = json.loads((source/'assets/psrl-staleness/animation-data.json').read_text(encoding='utf-8'))
    layers = set()
    for index, stage in enumerate(scene['stages'], 1):
        layers.update(f'bg-{index:02d}-{kind}.png' for kind in ('motion', 'still'))
        for names in stage['text_layers'].values():
            layers.update(names.values())
    layers.update(f'{name}-{state}.png' for name in scene['tracks'] for state in ('RESERVED', 'OCCUPIED'))
    layers.update(f'label-{label}.png' for label in ('待准入', '生成中', '已完成', '已训练', '训练中'))
    for name in sorted(layers):
        relative = 'animation-layers/' + name
        original = source/'assets/psrl-staleness'/relative
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        copied[relative] = sha256(original)
    html = (source/'publishing/psrl-staleness/animation-preview.html').read_text(encoding='utf-8')
    old_base = "const base='../../assets/psrl-staleness/animation-layers/';"
    if html.count(old_base) != 1:
        raise ValueError('Animation layer base changed; review the importer before copying.')
    html = html.replace(old_base, "const base='../animation-layers/';")
    target = destination/'animation/index.html'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html, encoding='utf-8')


def import_post(source, slug):
    settings = POSTS[slug]
    original_path = source/settings['source']
    original = original_path.read_text(encoding='utf-8')
    title = original.splitlines()[0].removeprefix('# ')
    destination = ROOT/'assets/blog'/slug
    destination.mkdir(parents=True, exist_ok=True)
    copied = {}
    figures = re.findall(r'!\[[^\]]*\]\((assets/[^)]+)\)', original)
    for figure in sorted(set(figures)):
        relative = figure.removeprefix(settings['asset_prefix'])
        if relative == figure:
            raise ValueError(f'Unexpected asset location: {figure}')
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/figure, target)
        copied[relative] = sha256(source/figure)
        if slug == 'async-rl-staleness' and Path(figure).suffix == '.svg':
            png = Path(figure).with_suffix('.png')
            shutil.copy2(source/png, target.with_suffix('.png'))
            copied[str(Path(relative).with_suffix('.png'))] = sha256(source/png)
    if slug == 'async-rl-staleness':
        with Image.open(destination/'systems-layer-map.png') as image:
            scale = image.width/1280
            cover = image.crop(tuple(round(value*scale) for value in (48, 216, 1232, 1061)))
            cover.save(destination/'cover.webp', quality=94)
            width, height = cover.size
    else:
        for extension in ('svg', 'png'):
            cover = source/f'assets/psrl-staleness/protocol-cover.{extension}'
            shutil.copy2(cover, destination/f'cover.{extension}')
            copied[f'cover.{extension}'] = sha256(cover)
        width, height = 1280, 720
        import_animation(source, destination, copied)
    metadata = {
        'layout': 'blog-post', 'title': title, 'description': settings['description'],
        'date': settings['date'], 'lang': 'zh-CN', 'series': settings['series'],
        'permalink': f'/blog/{slug}/',
        'cover': f'/assets/blog/{slug}/{settings["cover"]}?v={sha256(destination/settings["cover"])[:12]}',
        'cover_alt': settings['cover_alt'], 'cover_width': width, 'cover_height': height,
        'author_profile': False,
    }
    frontmatter = '\n'.join(key + ': ' + json.dumps(value, ensure_ascii=False) for key, value in metadata.items())
    (ROOT/'_blog').mkdir(exist_ok=True)
    (ROOT/f'_blog/{slug}.md').write_text('---\n' + frontmatter + '\n---\n\n' + adapt_article(original, slug), encoding='utf-8')
    manifest = {'source': settings['source'], 'source_sha256': sha256(original_path),
                'figures': len(figures), 'copied_assets': copied}
    (destination/'import-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'Imported {slug}: {len(figures)} figures, {len(copied)} source assets, and one cover.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Directory containing the original blogs')
    parser.add_argument('--article', choices=[*POSTS, 'all'], default='async-rl-staleness')
    args = parser.parse_args()
    for slug in POSTS if args.article == 'all' else [args.article]:
        import_post(args.source.resolve(), slug)


if __name__ == '__main__':
    main()
