from markdown_crawler import md_crawl
url = 'https://www.destateparks.com'
print(f'🕸️ Starting crawl of {url}')
md_crawl(
    url,
    max_depth=3,
    num_threads=2,
    base_dir='../output-markdown-2',
    valid_paths=['/'],
    target_content=['div.main-container', 'article', 'div.content', '#content', 'div.main-content'],  # Try multiple selectors
    is_domain_match=True,
    is_base_path_match=False,
    is_debug=True
)