"""Render the canonical Markdown reports without altering LaTeX contents.

For Piazza, supply the map of image URLs obtained from the editor's upload UI.
This utility prepares HTML only; publishing remains a browser operation.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
from pathlib import Path

import markdown
from bs4 import BeautifulSoup


def render(source: str, source_path: Path, uploaded_images: dict[str, str] | None = None,
           draft_missing_images: bool = False):
    equations = []

    def protect(match):
        equations.append(match.group(0))
        return f'MATHPLACEHOLDER{len(equations)-1}END'

    protected = re.sub(r'\$\$[\s\S]*?\$\$|\\\([\s\S]*?\\\)|(?<!\$)\$(?!\$)[^\n$]+?\$(?!\$)', protect, source)
    body = markdown.markdown(protected, extensions=['tables', 'fenced_code', 'sane_lists'])
    for index, equation in enumerate(equations):
        # Piazza recognizes display delimiters; preserve every TeX character.
        body = body.replace(f'MATHPLACEHOLDER{index}END', html.escape(equation))
    soup = BeautifulSoup(body, 'html.parser')
    images = []
    for index, img in enumerate(soup.find_all('img'), start=1):
        local = (source_path.parent / img['src']).resolve()
        previous_heading = img.find_previous('h2')
        images.append({
            'index': index,
            'name': local.name,
            'path': str(local),
            'alt': img.get('alt', ''),
            'section': previous_heading.get_text(' ', strip=True) if previous_heading else '',
        })
        if uploaded_images is None:
            img['src'] = local.as_uri()
        else:
            if local.name not in uploaded_images:
                if draft_missing_images:
                    placeholder = soup.new_tag('strong')
                    placeholder.string = f'【待上传图片：{img.get("alt", local.name)}；文件：{local.name}】'
                    img.replace_with(placeholder)
                    continue
                raise ValueError(f'Image has not been uploaded through Piazza editor: {local.name}')
            img['src'] = uploaded_images[local.name]
        img['style'] = 'max-width:100%;height:auto;'
    for table in soup.find_all('table'):
        table['style'] = 'border-collapse:collapse;width:100%;'
        for cell in table.find_all(['th', 'td']):
            cell['style'] = 'border:1px solid #b8c2cc;padding:7px;text-align:left;'
    return str(soup), {'equations': len(equations), 'images': images,
                       'headings': [h.get_text() for h in soup.find_all(re.compile('^h[1-6]$'))],
                       'tables': len(soup.find_all('table')), 'text': soup.get_text('\n')}


PAGE_STYLE = '''
body { max-width: 960px; margin: 2rem auto; padding: 0 1.25rem; color: #202124;
       font-family: system-ui, "Microsoft YaHei", sans-serif; line-height: 1.75; }
h1, h2, h3, h4 { line-height: 1.35; margin-top: 1.8em; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #b8c2cc; padding: 7px; text-align: left; vertical-align: top; }
img { max-width: 100%; height: auto; }
pre { overflow-x: auto; padding: 1rem; background: #f4f6f8; }
'''


def make_document(fragment: str, title: str) -> str:
    """Wrap a rendered Markdown fragment as a standalone UTF-8 HTML document."""
    return (
        '<!doctype html>\n'
        '<html lang="zh-CN">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{html.escape(title)}</title>\n'
        f'<style>{PAGE_STYLE}</style>\n'
        '</head>\n<body>\n'
        f'{fragment}\n'
        '</body>\n</html>\n'
    )


def plain_text(fragment: str) -> str:
    """Extract complete visible text and mark image insertion points for copy/paste."""
    soup = BeautifulSoup(fragment, 'html.parser')
    for img in soup.find_all('img'):
        alt = img.get('alt', '').strip()
        label = f'[图片插入点：{alt}]' if alt else '[图片插入点]'
        img.replace_with(label)
    return '\n'.join(line.rstrip() for line in soup.get_text('\n').split('\n')).strip('\n') + '\n'


def split_at_heading(fragment: str, level: int) -> list[str]:
    """Split at complete heading boundaries while retaining all intervening nodes."""
    soup = BeautifulSoup(fragment, 'html.parser')
    heading_name = f'h{level}'
    groups: list[list[object]] = []
    current: list[object] = []
    saw_first_heading = False
    for node in list(soup.contents):
        if getattr(node, 'name', None) == heading_name:
            if saw_first_heading and current:
                groups.append(current)
                current = [node]
            else:
                # Keep the report title and any preamble with the first chapter.
                current.append(node)
                saw_first_heading = True
        else:
            current.append(node)
    if current:
        groups.append(current)
    return [''.join(str(node) for node in group) for group in groups]


def build_parts(fragment: str, title: str, output: Path, limit: int) -> list[dict]:
    """Write complete section-aligned HTML/plain-text chunks when a page is large."""
    whole_document = make_document(fragment, title)
    if len(whole_document.encode('utf-8')) <= limit:
        text_path = output.with_suffix('.txt')
        text_path.write_text(plain_text(fragment), encoding='utf-8')
        return [{
            'file': output.name,
            'plain_text_file': text_path.name,
            'heading': title,
            'html_bytes': len(whole_document.encode('utf-8')),
            'plain_text_characters': len(plain_text(fragment)),
            'over_limit': False,
        }]

    sections = split_at_heading(fragment, 2)
    candidates: list[tuple[str, str]] = []
    for section in sections:
        section_soup = BeautifulSoup(section, 'html.parser')
        heading = section_soup.find(re.compile('^h[1-6]$'))
        section_title = heading.get_text(' ', strip=True) if heading else title
        section_document = make_document(section, f'{title} — {section_title}')
        if len(section_document.encode('utf-8')) <= limit:
            candidates.append((section_title, section))
            continue

        # If one second-level section is still too large, split only at complete
        # third-level headings. An indivisible oversized section is preserved whole.
        subsections = split_at_heading(section, 3)
        if len(subsections) > 1:
            for subsection in subsections:
                subsection_soup = BeautifulSoup(subsection, 'html.parser')
                subheading = subsection_soup.find(re.compile('^h[1-6]$'))
                subheading_title = subheading.get_text(' ', strip=True) if subheading else section_title
                candidates.append((f'{section_title} / {subheading_title}', subsection))
        else:
            candidates.append((section_title, section))

    parts = []
    total = len(candidates)
    for index, (heading, part_fragment) in enumerate(candidates, start=1):
        part_html = make_document(part_fragment, f'{title} — {heading}')
        part_text = plain_text(part_fragment)
        html_path = output.with_name(f'{output.stem}.part-{index:02d}.html')
        text_path = output.with_name(f'{output.stem}.part-{index:02d}.txt')
        html_path.write_text(part_html, encoding='utf-8')
        text_path.write_text(part_text, encoding='utf-8')
        byte_count = len(part_html.encode('utf-8'))
        parts.append({
            'part': index,
            'total_parts': total,
            'file': html_path.name,
            'plain_text_file': text_path.name,
            'heading': heading,
            'html_bytes': byte_count,
            'plain_text_characters': len(part_text),
            'over_limit': byte_count > limit,
        })
    return parts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image-map', type=Path)
    parser.add_argument('--draft-missing-images', action='store_true',
                        help='Leave explicit upload placeholders in an unpublished draft')
    parser.add_argument('--part-limit', type=int, default=50000,
                        help='UTF-8 byte limit before section-aligned HTML chunks are written')
    args = parser.parse_args()
    mapping = json.loads(args.image_map.read_text(encoding='utf-8')) if args.image_map else None
    fragment, manifest = render(args.source.read_text(encoding='utf-8'), args.source, mapping,
                                args.draft_missing_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if mapping is None:
        portable = BeautifulSoup(fragment, 'html.parser')
        for img, metadata in zip(portable.find_all('img'), manifest['images']):
            img['src'] = Path(os.path.relpath(metadata['path'], args.output.parent.resolve())).as_posix()
        fragment = str(portable)
    manifest['unpublished_draft'] = args.draft_missing_images
    title = manifest['headings'][0] if manifest['headings'] else args.source.stem
    document = make_document(fragment, title)
    args.output.write_text(document, encoding='utf-8')
    full_text = plain_text(fragment)
    text_path = args.output.with_suffix('.txt')
    text_path.write_text(full_text, encoding='utf-8')
    parts = build_parts(fragment, title, args.output, args.part_limit)
    # Compare all visible non-whitespace characters to ensure section splitting
    # neither loses nor duplicates text. Image insertion markers are included.
    compact = lambda value: re.sub(r'\s+', '', value)
    part_texts = [
        (args.output.parent / part['plain_text_file']).read_text(encoding='utf-8')
        for part in parts
    ]
    joined_parts = ''.join(part_texts)
    manifest['text'] = full_text
    manifest['plain_text_file'] = text_path.name
    manifest['parts'] = parts
    manifest['part_limit_bytes'] = args.part_limit
    manifest['parts_text_matches_full'] = compact(joined_parts) == compact(full_text)
    if not manifest['parts_text_matches_full']:
        raise RuntimeError('Section chunks do not reconstruct the full report text')
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'text'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
