import os

try:
    import markdown
except ImportError:
    markdown = None


def convert_md_to_html():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    md_path = os.path.join(current_dir, "QUANT_HANDBOOK.md")
    html_path = os.path.join(current_dir, "QUANT_HANDBOOK.html")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    if markdown:
        body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
    else:
        # 簡易 Markdown 解析備援
        import re

        body = md_text
        body = re.sub(r"^# (.*?)$", r"<h1>\1</h1>", body, flags=re.MULTILINE)
        body = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", body, flags=re.MULTILINE)
        body = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", body, flags=re.MULTILINE)
        body = re.sub(r"^#### (.*?)$", r"<h4>\1</h4>", body, flags=re.MULTILINE)
        body = body.replace("\n\n", "<br><br>")

    html_content = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量化金融與策略研發全方位核心講義 (Quant Handbook)</title>
    <!-- KaTeX 数学公式渲染引擎 -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
            onload="renderMathInElement(document.body, {{delimiters: [
                {{left: '$$', right: '$$', display: true}},
                {{left: '$', right: '$', display: false}}
            ]}});"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f8fafc;
            color: #0f172a;
            line-height: 1.8;
            margin: 0;
            padding: 2.5rem 1rem;
        }}
        .handbook-wrapper {{
            max-width: 960px;
            margin: 0 auto;
            background: #ffffff;
            padding: 3.5rem 3rem;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
        }}
        .print-btn {{
            position: fixed;
            bottom: 2rem;
            right: 2rem;
            background: #2563eb;
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 9999px;
            text-decoration: none;
            font-weight: 600;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.39);
            border: none;
            cursor: pointer;
            transition: all 0.2s;
            z-index: 100;
        }}
        .print-btn:hover {{ background: #1d4ed8; transform: translateY(-2px); }}
        h1 {{
            font-size: 2.25rem;
            color: #0f172a;
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 0.75rem;
            margin-bottom: 2rem;
        }}
        h2 {{
            font-size: 1.65rem;
            color: #1e3a8a;
            border-bottom: 1px solid #cbd5e1;
            padding-bottom: 0.5rem;
            margin-top: 3rem;
        }}
        h3 {{ font-size: 1.25rem; color: #1d4ed8; margin-top: 2rem; }}
        h4 {{ font-size: 1.1rem; color: #334155; margin-top: 1.5rem; }}
        code {{
            background: #f1f5f9;
            color: #e11d48;
            padding: 0.2rem 0.45rem;
            border-radius: 6px;
            font-size: 0.9em;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}
        pre {{
            background: #0f172a;
            color: #f8fafc;
            padding: 1.25rem;
            border-radius: 10px;
            overflow-x: auto;
            font-size: 0.9em;
        }}
        pre code {{ background: transparent; color: inherit; padding: 0; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 2rem 0;
            font-size: 0.95em;
        }}
        th, td {{
            border: 1px solid #cbd5e1;
            padding: 0.85rem 1.2rem;
            text-align: left;
        }}
        th {{ background-color: #f1f5f9; font-weight: 700; color: #1e293b; }}
        blockquote {{
            border-left: 5px solid #3b82f6;
            margin: 1.75rem 0;
            padding: 0.75rem 1.5rem;
            background-color: #eff6ff;
            color: #1e40af;
            border-radius: 0 8px 8px 0;
        }}
        a {{ color: #2563eb; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        @media (max-width: 768px) {{
            .handbook-wrapper {{ padding: 1.5rem 1.2rem; border-radius: 0; }}
            body {{ padding: 0; }}
        }}
        @media print {{
            body {{ background: white; padding: 0; }}
            .handbook-wrapper {{ box-shadow: none; border: none; max-width: 100%; padding: 0; }}
            .print-btn {{ display: none; }}
        }}
    </style>
</head>
<body>
    <button class="print-btn" onclick="window.print()">🖨️ 列印 / 另存為 PDF</button>
    <div class="handbook-wrapper">
        {body}
    </div>
</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[✓] 講義 HTML 已生成: {html_path}")


if __name__ == "__main__":
    convert_md_to_html()
