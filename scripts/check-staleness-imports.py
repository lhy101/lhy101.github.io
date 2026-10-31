"""Verify imported prose, formulas, metadata, assets, and player dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from import_staleness_articles import POSTS, ROOT, adapt_article


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    report = []
    for slug, settings in POSTS.items():
        source = args.source/settings['source']
        original = source.read_text(encoding='utf-8')
        imported = (ROOT/f'_blog/{slug}.md').read_text(encoding='utf-8')
        _, metadata, body = imported.split('---\n', 2)
        assert body.removeprefix('\n') == adapt_article(original, slug), slug+' body parity'
        destination = ROOT/'assets/blog'/slug
        manifest = json.loads((destination/'import-manifest.json').read_text(encoding='utf-8'))
        assert manifest['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
        for relative, expected in manifest['copied_assets'].items():
            assert hashlib.sha256((destination/relative).read_bytes()).hexdigest() == expected, relative
        for match in re.finditer(r"\{\{ '/assets/blog/"+slug+r"/([^']+)' \| relative_url \}\}", body):
            assert (destination/match[1].split('?')[0]).is_file(), match[1]
        assert f'permalink: "/blog/{slug}/"' in metadata
        assert f'date: "{settings["date"]}"' in metadata
        assert re.findall(r'(?m)^#{1,6} .+$', original) == re.findall(r'(?m)^#{1,6} .+$', body)
        strip_equations = lambda text: re.sub(r'\\(?:begin|end)\{equation\*\}', '', text)
        formulas = lambda text: [re.sub(r'\s+', '', strip_equations(m)) for m in re.findall(r'\$\$([\s\S]*?)\$\$', text)]
        assert formulas(original) == formulas(body), slug+' formula parity'
        assert len(re.findall('<details', original)) == len(re.findall('<details markdown="1"', body))
        if slug == 'psrl-staleness':
            assert manifest['figures'] == 23
            assert body.count('<details markdown="1" class="protocol-detail"') == 7
            player = (destination/'animation/index.html').read_text(encoding='utf-8')
            assert "const base='../animation-layers/';" in player
            assert '../../assets/psrl-staleness/' not in player
            assert '<text ' not in (destination/'cover.svg').read_text(encoding='utf-8')
        report.append({'slug': slug, 'figures': manifest['figures'],
                       'verified_assets': len(manifest['copied_assets']), 'prose_and_formulas': 'unchanged'})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
