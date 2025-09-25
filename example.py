from markdown_crawler import md_crawl
url = 'https://www.destateparks.com'
print(f'🕸️ Starting crawl of {url}')
md_crawl(
    url,
    max_depth=1,
    num_threads=5,
    base_dir='../output-markdown-depth1-new',
    valid_paths=['/'],
    target_content=['div.main-container', 'article', 'div.content', '#content', 'div.main-content'],  # Try multiple selectors
    is_domain_match=True,
    is_base_path_match=False,
    is_debug=True
)